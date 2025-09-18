import copy
import logging
import numpy as np
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from ood import MSPDetector, EnergyDetector, ODINDetector, LTSIndividualDetector, LTSFusionDetector, LTSRGBOnlyDetector, LTSLateFusionDetector, LTSRGBOnlyNoNormDetector, LTSGyroOnlyDetector, LTSAcceOnlyDetector
from ood.metrics import compute_ood_metrics, compute_threshold_accuracy


EPSILON = 1e-8
batch_size = 64


class MMEABaseLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self._batch_size = args["batch_size"]
        self._num_workers = args["workers"]
        self._lr = args["lr"]
        self._epochs = args["epochs"]
        self._momentum = args["momentum"]
        self._weight_decay = args["weight_decay"]
        self._lr_steps = args["lr_steps"]
        self._modality = args["modality"]

        self._partialbn = args["partialbn"]
        self._freeze = args["freeze"]
        self._clip_gradient = args["clip_gradient"]
        self.enable_ood = args["enable_ood"]


        self.fisher = None
        self._network = None # Placeholder for the network
        self.class_increments = []

    def _setup_data_loaders_with_ood(self, data_manager):
        """Setup train/test/ood data loaders based on enable_ood setting"""
        logging.info(f"Setting up data loaders for Task {self._cur_task}")
        
        # Training data: current task classes only
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(), # return None, if memory_size is 0
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers
        )
        
        # Test data: all seen classes so far  
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), 
            source="test", 
            mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
        )
        
        # OOD Test data (only if OOD is enabled)
        self.ood_test_loader = None
        if self.enable_ood:
            if self._total_classes < self.total_classnum:
                ood_test_dataset = data_manager.get_dataset(
                    np.arange(self._total_classes, self.total_classnum),
                    source="test",
                    mode="test",
                )
                self.ood_test_loader = DataLoader(
                    ood_test_dataset,
                    batch_size=self._batch_size,
                    shuffle=False,
                    num_workers=self._num_workers,
                )
                logging.info(f"  ✅ OOD enabled. OOD classes: {self._total_classes} ~ {self.total_classnum-1}")
                logging.info(f"  📊 OOD test samples: {len(ood_test_dataset)}")
            else:
                logging.info("  ⚠️ OOD enabled, but no unseen classes remain (final task).")
        else:
            logging.info("  ❌ OOD disabled (enable_ood=False). Skipping OOD loader creation.")

        logging.info(f"  📚 Train samples: {len(train_dataset)}")
        logging.info(f"  🧪 ID test samples: {len(test_dataset)}")
    
    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        optimizer = self._choose_optimizer()

        # Setup scheduler
        if type(optimizer) == list:
            scheduler_adam = optim.lr_scheduler.MultiStepLR(optimizer[0], self._lr_steps, gamma=0.1)
            scheduler_sgd = optim.lr_scheduler.MultiStepLR(optimizer[1], self._lr_steps, gamma=0.1)
            scheduler = [scheduler_adam, scheduler_sgd]
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, self._lr_steps, gamma=0.1)

        if self._cur_task == 0:
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        schedulers = scheduler if isinstance(scheduler, (list, tuple)) else [scheduler]
        
        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
                
            if self._partialbn:
                self._network.backbone.freeze_fn('partialbn_statistics')
            if self._freeze:
                self._network.backbone.freeze()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                if self.args["debug_mode"] and i >= 5:
                    break
                
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)

                # zero gradients
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)

                loss.backward()

                if self._clip_gradient is not None:
                    total_norm = nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)

                # optimizer step
                for opt in optimizers:
                    opt.step()

                losses += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.numel()

            # epoch-level scheduler step
            for sch in schedulers:
                sch.step()

            train_acc = round((correct * 100.0) / max(1, total), 2)

            # Log training metrics to W&B
            if self.args['use_wandb']:
                wandb.log({
                    "Train/train_loss": losses / len(train_loader),
                    "Train/train_accuracy": train_acc,
                })

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self._epochs,
                losses / len(train_loader),
                train_acc,
            )
            if self.args.get("log_test_acc", False) and epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
                # Log test metrics to W&B
                if self.args['use_wandb']:
                    wandb.log({
                        "Train/test_accuracy": test_acc
                    })
            
            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        pass

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            for m in self._modality:
                inputs[m] = inputs[m].to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            for m in self._modality:
                inputs[m] = inputs[m].to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
    def evaluate_cl(self):
        """Evaluate only CL accuracy"""
        logging.info(f"=== Task {self._cur_task} CL Evaluation ===")
        logging.info(f"Known classes: 0-{self._classes_seen_so_far-1}")
        
        # Standard CL accuracy evaluation
        cnn_accy, nme_accy = self.eval_task()
        
        if nme_accy is not None:
            logging.info(f"CL Accuracy - CNN: {cnn_accy['top1']:.2f}%, NME: {nme_accy['top1']:.2f}%")
        else:
            logging.info(f"CL Accuracy - CNN: {cnn_accy['top1']:.2f}%, NME: Not Available")
            
        # Log task metrics to W&B (모든 CL 메트릭을 한 번에 로깅)
        if self.args.get('use_wandb', False):
            cl_metrics = {"Task/avg_acc": cnn_accy['top1']}
            
            # CL grouped accuracy 추가
            for k, v in cnn_accy['grouped'].items():
                cl_metrics[f"Task/[{k}]_acc"] = v

            # NME accuracy 추가 (있는 경우)
            if nme_accy is not None:
                cl_metrics["Task/nme_avg_acc"] = nme_accy['top1']
                for k, v in nme_accy.get('grouped', {}).items():
                    cl_metrics[f"Task/NME_[{k}]_acc"] = v
            
            # 🎯 모든 CL 메트릭을 한 번에 로깅 (동일한 step)
            logging.info("📊 Logging all CL metrics to wandb in a single step...")
            wandb.log(cl_metrics)
            logging.info(f"✅ Logged {len(cl_metrics)} CL metrics to wandb")
        
        return {'cnn': cnn_accy, 'nme': nme_accy if nme_accy else {'top1': 0.0, 'grouped': {}}}

    def evaluate_ood(self):
        """Evaluate only OOD detection performance"""
        if not self.enable_ood:
            logging.info("OOD evaluation disabled (enable_ood=False).")
            return {}, {}
            
        logging.info(f"=== Task {self._cur_task} OOD Evaluation ===")
        logging.info(f"Known classes: 0-{self._classes_seen_so_far-1}")
        logging.info(f"Unknown classes: {self._classes_seen_so_far}-{self.total_classnum-1}")
        
        # Check OOD configuration
        if "ood_methods" not in self.args:
            logging.error("ood_methods not found in configuration file!")
            return {}, {}
                  
        ood_methods = self.args["ood_methods"]
        logging.info(f"OOD Methods: {ood_methods}")
        
        if self.ood_test_loader is None:
            logging.warning("No OOD test data available. Skipping OOD evaluation.")
            return {}, {}
        
        ood_results = {}
        score_distributions = {}
        all_wandb_metrics = {}  # 모든 OOD 메트릭을 저장할 딕셔너리
        
        logging.info("=== OOD Detection Results ===")
                
        # Check if individual features are needed
        need_individual_features = "LTS_Individual" in ood_methods
        need_fusion_features = "LTS_Fusion" in ood_methods
        
        # Extract data in single forward pass
        print("  📊 Processing ID data (logits + features)...")
        if need_individual_features:
            print("  🔍 LTS_Individual detected - also extracting individual modality features...")
        if need_fusion_features:
            print("  🔍 LTS_Fusion detected - also extracting fusion features...")
        
        id_data = self._extract_data_batch(
            self.test_loader, 
            extract_features=True, 
            extract_logits=True,
            extract_individual_features=need_individual_features
        )
        id_logits = id_data['logits']
        id_features = id_data['features'] 
        id_labels = id_data['labels']
        id_individual_features = id_data['individual_features']
        
        print("  🎯 Processing OOD data (logits + features)...")
        ood_data = self._extract_data_batch(
            self.ood_test_loader, 
            extract_features=True, 
            extract_logits=True,
            extract_individual_features=need_individual_features
        )
        ood_logits = ood_data['logits']
        ood_features = ood_data['features']
        ood_labels = ood_data['labels']
        ood_individual_features = ood_data['individual_features']
        
        print(f"✅ Data extracted - ID: logits{id_logits.shape}, features{id_features.shape}")
        print(f"                   OOD: logits{ood_logits.shape}, features{ood_features.shape}")
        if need_individual_features:
            print(f"                   Individual features: {len(id_individual_features)} modalities, shapes: {[f.shape for f in id_individual_features]}")
        
        # Store extracted data for visualization
        self._cached_id_data = {'features': id_features, 'labels': id_labels}
        self._cached_ood_data = {'features': ood_features, 'labels': ood_labels}

        for method_name in tqdm(ood_methods, desc="OOD Methods", position=0):
            try:
                # Initialize OOD detector
                if method_name == "MSP":
                    detector = MSPDetector(self._network, self._device)
                elif method_name == "Energy":
                    detector = EnergyDetector(self._network, self._device)
                elif method_name == "ODIN":
                    detector = ODINDetector(self._network, self._device)
                elif method_name == "LTS_Individual":
                    detector = LTSIndividualDetector(self._network, self._device)
                elif method_name == "LTS_Fusion":
                    detector = LTSFusionDetector(self._network, self._device)
                elif method_name == "LTS_RGB_Only":
                    detector = LTSRGBOnlyDetector(self._network, self._device)
                elif method_name == "LTS_Late_Fusion":
                    detector = LTSLateFusionDetector(self._network, self._device, fusion_method='weighted_average')
                elif method_name == "LTS_RGB_Only_No_Norm":
                    detector = LTSRGBOnlyNoNormDetector(self._network, self._device)
                elif method_name == "LTS_Gyro_Only":
                    detector = LTSGyroOnlyDetector(self._network, self._device)
                elif method_name == "LTS_Acce_Only":
                    detector = LTSAcceOnlyDetector(self._network, self._device)
                else:
                    logging.warning(f"Unknown OOD method: {method_name}")
                    continue
                
                logging.info(f"Computing {method_name} scores...")
                
                # Special handling for LTS_Individual method (needs individual features)
                if method_name == "LTS_Individual":
                    if id_individual_features is None or ood_individual_features is None:
                        logging.error("LTS_Individual method requires individual features, but they were not extracted!")
                        continue
                    
                    logging.info(f"  🔍 LTS_Individual processing:")
                    logging.info(f"    - ID logits: {id_logits.shape}")
                    logging.info(f"    - OOD logits: {ood_logits.shape}")
                    logging.info(f"    - Individual features: {len(id_individual_features)} modalities")
                    for i, feat in enumerate(id_individual_features):
                        logging.info(f"      Modality {i}: {feat.shape}")
                    
                    # Compute scores using pre-extracted individual features
                    id_scores = detector.compute_scores_with_features(id_logits, id_individual_features)
                    ood_scores = detector.compute_scores_with_features(ood_logits, ood_individual_features)
                    
                    logging.info(f"  ✅ LTS_Individual scores computed: ID={len(id_scores)}, OOD={len(ood_scores)}")
                elif method_name == "LTS_Fusion":
                    if id_features is None or ood_features is None:
                        logging.error("LTS_Fusion method requires fusion features, but they were not extracted!")
                        continue
                    
                    logging.info(f"  🔍 LTS_Fusion processing:")
                    logging.info(f"    - ID logits: {id_logits.shape}")
                    logging.info(f"    - OOD logits: {ood_logits.shape}")
                    logging.info(f"    - ID fusion features: {id_features.shape}")
                    logging.info(f"    - OOD fusion features: {ood_features.shape}")
                    
                    # Convert numpy arrays back to tensors for LTS_Fusion
                    id_features_tensor = torch.from_numpy(id_features).to(self._device)
                    ood_features_tensor = torch.from_numpy(ood_features).to(self._device)
                    
                    # Compute scores using fusion features
                    id_scores = detector.compute_scores_with_fusion_features(id_logits, id_features_tensor)
                    ood_scores = detector.compute_scores_with_fusion_features(ood_logits, ood_features_tensor)
                    
                    logging.info(f"  ✅ LTS_Fusion scores computed: ID={len(id_scores)}, OOD={len(ood_scores)}")
                elif method_name == "LTS_RGB_Only":
                    if id_individual_features is None or ood_individual_features is None:
                        logging.error("LTS_RGB_Only method requires individual features, but they were not extracted!")
                        continue
                    
                    # Check if RGB modality is available
                    if "RGB" not in self._modality:
                        logging.warning(f"  ⚠️  LTS_RGB_Only skipped - RGB modality not available in {self._modality}")
                        continue
                    
                    logging.info(f"  🔍 LTS_RGB_Only processing:")
                    logging.info(f"    - ID logits: {id_logits.shape}")
                    logging.info(f"    - OOD logits: {ood_logits.shape}")
                    logging.info(f"    - Model modalities: {self._modality}")
                    logging.info(f"    - Using RGB modality only (index {self._modality.index('RGB')})")
                    rgb_idx = self._modality.index('RGB')
                    logging.info(f"    - RGB features shape: ID={id_individual_features[rgb_idx].shape}, OOD={ood_individual_features[rgb_idx].shape}")
                    
                    # Create feature lists containing only RGB features
                    rgb_id_features = [id_individual_features[rgb_idx]]
                    rgb_ood_features = [ood_individual_features[rgb_idx]]
                    
                    # Compute scores using RGB features only
                    id_scores = detector.compute_scores_with_features(id_logits, rgb_id_features)
                    ood_scores = detector.compute_scores_with_features(ood_logits, rgb_ood_features)
                    
                    logging.info(f"  ✅ LTS_RGB_Only scores computed: ID={len(id_scores)}, OOD={len(ood_scores)}")
                elif method_name == "LTS_Late_Fusion":
                    if id_individual_features is None or ood_individual_features is None:
                        logging.error("LTS_Late_Fusion method requires individual features, but they were not extracted!")
                        continue
                    
                    logging.info(f"  🔍 LTS_Late_Fusion processing:")
                    logging.info(f"    - ID logits: {id_logits.shape}")
                    logging.info(f"    - OOD logits: {ood_logits.shape}")
                    logging.info(f"    - Individual features: {len(id_individual_features)} modalities")
                    logging.info(f"    - Fusion method: weighted_average")
                    
                    # Compute scores using late fusion approach
                    id_scores = detector.compute_scores_with_features(id_logits, id_individual_features)
                    ood_scores = detector.compute_scores_with_features(ood_logits, ood_individual_features)
                    
                    # Check if method returned None (not applicable)
                    if id_scores is None or ood_scores is None:
                        logging.warning(f"  ⚠️  LTS_Late_Fusion returned None - skipping this method")
                        continue
                    
                    logging.info(f"  ✅ LTS_Late_Fusion scores computed: ID={len(id_scores)}, OOD={len(ood_scores)}")
                elif method_name == "LTS_RGB_Only_No_Norm":
                    if id_individual_features is None or ood_individual_features is None:
                        logging.error("LTS_RGB_Only_No_Norm method requires individual features, but they were not extracted!")
                        continue
                    
                    logging.info(f"  🔍 LTS_RGB_Only_No_Norm processing:")
                    logging.info(f"    - ID logits: {id_logits.shape}")
                    logging.info(f"    - OOD logits: {ood_logits.shape}")
                    logging.info(f"    - Using RGB modality only (index 0) WITHOUT L2 normalization")
                    logging.info(f"    - RGB features shape: ID={id_individual_features[0].shape}, OOD={ood_individual_features[0].shape}")
                    
                    # Compute scores using RGB features only (no normalization)
                    id_scores = detector.compute_scores_with_features(id_logits, id_individual_features)
                    ood_scores = detector.compute_scores_with_features(ood_logits, ood_individual_features)
                    
                    logging.info(f"  ✅ LTS_RGB_Only_No_Norm scores computed: ID={len(id_scores)}, OOD={len(ood_scores)}")
                elif method_name == "LTS_Gyro_Only":
                    if id_individual_features is None or ood_individual_features is None:
                        logging.error("LTS_Gyro_Only method requires individual features, but they were not extracted!")
                        continue
                    
                    # Check if Gyro modality is available
                    if "Gyro" not in self._modality:
                        logging.warning(f"  ⚠️  LTS_Gyro_Only skipped - Gyro modality not available in {self._modality}")
                        continue
                    
                    logging.info(f"  🔍 LTS_Gyro_Only processing:")
                    logging.info(f"    - ID logits: {id_logits.shape}")
                    logging.info(f"    - OOD logits: {ood_logits.shape}")
                    logging.info(f"    - Model modalities: {self._modality}")
                    gyro_idx = self._modality.index('Gyro')
                    logging.info(f"    - Using Gyro modality only (index {gyro_idx})")
                    logging.info(f"    - Gyro features shape: ID={id_individual_features[gyro_idx].shape}, OOD={ood_individual_features[gyro_idx].shape}")
                    
                    # Create feature lists containing only Gyro features
                    gyro_id_features = [id_individual_features[gyro_idx]]
                    gyro_ood_features = [ood_individual_features[gyro_idx]]
                    
                    # Compute scores using Gyro features only
                    id_scores = detector.compute_scores_with_features(id_logits, gyro_id_features)
                    ood_scores = detector.compute_scores_with_features(ood_logits, gyro_ood_features)
                    
                    # Check if method returned None (not applicable)
                    if id_scores is None or ood_scores is None:
                        logging.warning(f"  ⚠️  LTS_Gyro_Only returned None - skipping this method")
                        continue
                    
                    logging.info(f"  ✅ LTS_Gyro_Only scores computed: ID={len(id_scores)}, OOD={len(ood_scores)}")
                elif method_name == "LTS_Acce_Only":
                    if id_individual_features is None or ood_individual_features is None:
                        logging.error("LTS_Acce_Only method requires individual features, but they were not extracted!")
                        continue
                    
                    # Check if Accelerometer modality is available
                    if "Acce" not in self._modality:
                        logging.warning(f"  ⚠️  LTS_Acce_Only skipped - Acce modality not available in {self._modality}")
                        continue
                    
                    logging.info(f"  🔍 LTS_Acce_Only processing:")
                    logging.info(f"    - ID logits: {id_logits.shape}")
                    logging.info(f"    - OOD logits: {ood_logits.shape}")
                    logging.info(f"    - Model modalities: {self._modality}")
                    acce_idx = self._modality.index('Acce')
                    logging.info(f"    - Using Accelerometer modality only (index {acce_idx})")
                    logging.info(f"    - Accelerometer features shape: ID={id_individual_features[acce_idx].shape}, OOD={ood_individual_features[acce_idx].shape}")
                    
                    # Create feature lists containing only Accelerometer features
                    acce_id_features = [id_individual_features[acce_idx]]
                    acce_ood_features = [ood_individual_features[acce_idx]]
                    
                    # Compute scores using Accelerometer features only
                    id_scores = detector.compute_scores_with_features(id_logits, acce_id_features)
                    ood_scores = detector.compute_scores_with_features(ood_logits, acce_ood_features)
                    
                    # Check if method returned None (not applicable)
                    if id_scores is None or ood_scores is None:
                        logging.warning(f"  ⚠️  LTS_Acce_Only returned None - skipping this method")
                        continue
                    
                    logging.info(f"  ✅ LTS_Acce_Only scores computed: ID={len(id_scores)}, OOD={len(ood_scores)}")
                else:
                    # Regular methods using only logits
                    id_scores = detector.compute_scores_from_cached_logits(id_logits)      
                    ood_scores = detector.compute_scores_from_cached_logits(ood_logits) 
                
                # Store score distributions for visualization
                score_distributions[method_name] = {
                    'id_scores': id_scores.tolist() if hasattr(id_scores, 'tolist') else list(id_scores),
                    'ood_scores': ood_scores.tolist() if hasattr(ood_scores, 'tolist') else list(ood_scores)
                }
                
                # Compute OOD metrics
                metrics = compute_ood_metrics(id_scores, ood_scores, method_name)
                ood_results[method_name] = metrics
                
                # Log results
                if 'error' not in metrics:
                    cm_fpr95 = metrics['confusion_fpr95']
                    cm_youden = metrics['confusion_youden']
                    
                    logging.info(f"{method_name}: AUROC={metrics['auroc']:.2f}%, AUPR_ID={metrics['aupr_id']:.2f}%, AUPR_OOD={metrics['aupr_ood']:.2f}%")
                    logging.info(f"  CM@FPR95: TP={cm_fpr95['tp']} FP={cm_fpr95['fp']} TN={cm_fpr95['tn']} FN={cm_fpr95['fn']} | TPR={cm_fpr95['tpr']:.3f} FPR={cm_fpr95['fpr']:.3f}")
                    logging.info(f"  CM@YoudenJ: TP={cm_youden['tp']} FP={cm_youden['fp']} TN={cm_youden['tn']} FN={cm_youden['fn']} | YoudenJ={cm_youden['youdenJ']:.3f}")
                    
                    # 🔄 모든 OOD 메트릭을 하나의 딕셔너리에 수집 (개별 wandb.log 호출 대신)
                    if self.args.get('use_wandb', False):
                        method_metrics = {
                            # Core metrics
                            f"Task/{method_name}_auroc": metrics['auroc'],
                            f"Task/{method_name}_aupr_id": metrics['aupr_id'],
                            f"Task/{method_name}_aupr_ood": metrics['aupr_ood'],
                            
                            # FPR95 기준 Confusion Matrix
                            f"Task/{method_name}_fpr95_tp": cm_fpr95['tp'],
                            f"Task/{method_name}_fpr95_fp": cm_fpr95['fp'],
                            f"Task/{method_name}_fpr95_tn": cm_fpr95['tn'],
                            f"Task/{method_name}_fpr95_fn": cm_fpr95['fn'],
                            f"Task/{method_name}_fpr95_precision": cm_fpr95['precision'],
                            f"Task/{method_name}_fpr95_recall": cm_fpr95['recall'],
                            f"Task/{method_name}_fpr95_f1": cm_fpr95['f1'],
                            f"Task/{method_name}_fpr95_tpr": cm_fpr95['tpr'],
                            f"Task/{method_name}_fpr95_fpr": cm_fpr95['fpr'],
                            f"Task/{method_name}_fpr95_threshold": cm_fpr95['threshold'],
                            
                            # Youden's J 기준 Confusion Matrix
                            f"Task/{method_name}_youden_tp": cm_youden['tp'],
                            f"Task/{method_name}_youden_fp": cm_youden['fp'],
                            f"Task/{method_name}_youden_tn": cm_youden['tn'],
                            f"Task/{method_name}_youden_fn": cm_youden['fn'],
                            f"Task/{method_name}_youden_precision": cm_youden['precision'],
                            f"Task/{method_name}_youden_recall": cm_youden['recall'],
                            f"Task/{method_name}_youden_f1": cm_youden['f1'],
                            f"Task/{method_name}_youden_tpr": cm_youden['tpr'],
                            f"Task/{method_name}_youden_fpr": cm_youden['fpr'],
                            f"Task/{method_name}_youden_threshold": cm_youden['threshold'],
                            f"Task/{method_name}_youdenJ": cm_youden['youdenJ']
                        }
                        all_wandb_metrics.update(method_metrics)
                else:
                    logging.error(f"{method_name}: Error - {metrics['error']}")
                    
            except Exception as e:
                logging.error(f"{method_name} evaluation failed: {e}")
                ood_results[method_name] = {'error': str(e), 'method': method_name}
        
        # 🎯 모든 OOD 방법론의 결과를 한 번에 wandb에 로깅 (동일한 step)
        if self.args.get('use_wandb', False) and all_wandb_metrics:
            logging.info("📊 Logging all OOD metrics to wandb in a single step...")
            wandb.log(all_wandb_metrics)
            logging.info(f"✅ Logged {len(all_wandb_metrics)} OOD metrics to wandb")
        
        # Store results and data for visualization
        self.latest_ood_results = ood_results
        self._visualization_data = {
            'id_features': id_features,
            'id_labels': id_labels,
            'ood_features': ood_features,
            'score_distributions': score_distributions
        }
        
        return ood_results, score_distributions
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint for inference"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            self._network.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logging.info(f"✅ Model weights loaded successfully")
        else:
            logging.warning("No model_state_dict found in checkpoint")
        
        # Load task information if available
        if 'tasks' in checkpoint:
            self._cur_task = checkpoint['tasks']
            logging.info(f"✅ Task info loaded: current task = {self._cur_task}")
        
        # iCaRL: Load class means for NME evaluation if available
        if 'class_means' in checkpoint:
            self._class_means = checkpoint['class_means']
            logging.info(f"✅ Class means loaded for {len(self._class_means)} classes")
        else:
            logging.warning("⚠️  No class means found in checkpoint - NME evaluation will be unavailable")
        
        self._network.eval()
        return checkpoint

    def inference_mode_evaluation(self, data_manager, checkpoint_path, task_id):
        """Run evaluation using pre-trained checkpoint without training"""
        logging.info(f"=== Inference Mode: Task {task_id} ===")
        
        # Set task state for evaluation first (needed for correct classifier size)
        self._cur_task = task_id
        
        # Calculate total classes up to this task
        total_classes_for_task = 0
        for i in range(task_id + 1):
            total_classes_for_task += data_manager.get_task_size(i)
            
        self._total_classes = total_classes_for_task
        self._classes_seen_so_far = self._total_classes
        
        # Set total_classnum for OOD evaluation
        self.total_classnum = data_manager.get_total_classnum()
        
        # Update network classifier to match checkpoint size BEFORE loading
        self._update_classifier(self._total_classes)
        
        # Load checkpoint with correct classifier size
        self.load_checkpoint(checkpoint_path)
        
        # Ensure model is on correct device
        self._network = self._network.to(self._device)
        
        # Setup data loaders for evaluation
        self._setup_data_loaders_with_ood(data_manager)
        
        # 🔍 DEBUG: Check if test_loader is properly set
        if hasattr(self, 'test_loader') and self.test_loader is not None:
            logging.info(f"✅ test_loader properly set with {len(self.test_loader.dataset)} samples")
        else:
            logging.error("❌ test_loader is not set properly!")
            raise AttributeError("test_loader was not set up correctly")
        
        # Perform evaluation
        if self.enable_ood:
            # Perform both CL and OOD evaluation
            cl_results = self.evaluate_cl()
            ood_results, score_distributions = self.evaluate_ood()
            return cl_results, ood_results, score_distributions
        else:
            # Perform only CL evaluation
            cl_results = self.evaluate_cl()
            return cl_results
    
    def _update_classifier(self, nb_classes):
        """Update classifier based on network type"""
        if hasattr(self._network, 'update_fc'):
            # TBN: use update_fc method (total classes)
            self._network.update_fc(nb_classes)
        elif hasattr(self._network, 'gen_train_fc'):
            # TSN: use gen_train_fc method (incremental classes only)
            incre_classes = nb_classes - self._known_classes  
            self._network.gen_train_fc(incre_classes)
        else:
            raise NotImplementedError(f"Network {type(self._network)} doesn't support classifier update")
        
        # Run evaluations
        cl_results = self.evaluate_cl()
        
        if self.enable_ood:
            ood_results, score_distributions = self.evaluate_ood()
            return cl_results, ood_results, score_distributions
        else:
            return cl_results, {}, {}

    def clear_cached_data(self):
        """Clear cached data to free memory after T-SNE visualization"""
        if hasattr(self, '_cached_id_data'):
            del self._cached_id_data
        if hasattr(self, '_cached_ood_data'):
            del self._cached_ood_data
        logging.info("🧹 Cleared cached feature data to free memory")
    
    def _extract_data_batch(self, loader, extract_features=True, extract_logits=True, extract_individual_features=False):
        """
        통합된 데이터 추출 함수 - 한 번의 forward pass로 logits, features, labels, individual_features 추출
        
        Args:
            loader: DataLoader
            extract_features: Whether to extract fused features for T-SNE
            extract_logits: Whether to extract logits for OOD detection
            extract_individual_features: Whether to extract individual modality features for LTS
            
        Returns:
            dict: {'logits': tensor, 'features': array, 'labels': array, 'individual_features': list}
        """
        self._network.eval()
        all_logits = []
        all_features = []
        all_labels = []
        all_individual_features = []  # For individual modality features
        C_t = self._classes_seen_so_far  # 누적 클래스 수

        with torch.no_grad():
            for _, inputs, targets in tqdm(loader, desc="Extracting data", leave=False):
                if isinstance(inputs, dict):
                    for m in inputs:
                        inputs[m] = inputs[m].to(self._device)
                else:
                    inputs = inputs.to(self._device)

                # 한 번의 forward pass로 모든 데이터 추출
                try:
                    # 네트워크 타입에 따른 조건부 호출 (TSN vs TBN 호환성)
                    if hasattr(self._network, '__class__') and 'TSN' in self._network.__class__.__name__:
                        # TSNBaseline: mode 파라미터와 cur_task_size 지원
                        outputs = self._network(inputs, cur_task_size=C_t, mode='test')
                    else:
                        # TBNBaseline: 기본 forward 사용
                        outputs = self._network(inputs)
                    
                    # 1. Logits 추출
                    if extract_logits:
                        all_logits.append(outputs["logits"].cpu())
                    
                    # 2. Features 추출 (fusion 출력)
                    if extract_features:
                        features = None
                        if 'fusion_features' in outputs:
                            features = outputs['fusion_features']  # TSN
                        elif 'features' in outputs:
                            features = outputs['features']  # TBN
                        else:
                            # fallback: extract_vector 사용
                            features = self._network.extract_vector(inputs)
                        
                        # Ensure features are 2D [batch_size, feature_dim]
                        if features is not None:
                            if features.dim() > 2:
                                features = features.view(features.size(0), -1)
                            all_features.append(features.cpu())
                        else:
                            # Skip batch if features extraction failed
                            logging.warning(f"Features extraction failed for batch, skipping...")
                            continue
                    
                    # 3. Individual features 추출 (LTS용)
                    if extract_individual_features:
                        # 한 번에 전체 배치의 individual features 추출
                        individual_features_batch = self._network.backbone(inputs)
                        
                        # Convert to list if needed
                        if not isinstance(individual_features_batch, list):
                            individual_features_batch = [individual_features_batch]
                        
                        # Flatten features for each modality and handle segments
                        flattened_features = []
                        batch_size = targets.size(0)  # 실제 배치 크기
                        
                        for feat in individual_features_batch:
                            if feat.dim() > 2:
                                feat = feat.view(feat.size(0), -1)  # (segments*batch_size, feature_dim)
                            
                            # TBN의 경우 segments별로 나누어 평균내기
                            if feat.size(0) != batch_size:
                                # segments가 있는 경우: (segments*batch_size, feature_dim) -> (batch_size, feature_dim)
                                num_segments = feat.size(0) // batch_size
                                feat = feat.view(batch_size, num_segments, -1).mean(dim=1)  # 평균내기
                            
                            flattened_features.append(feat.cpu())
                        
                        all_individual_features.append(flattened_features)
                    
                    all_labels.append(targets.cpu())
                    
                except Exception as e:
                    logging.warning(f"Data extraction failed for batch: {e}")
                    logging.warning(f"Batch targets shape: {targets.shape}, inputs type: {type(inputs)}")
                    logging.warning(f"Skipping this batch to avoid dummy data contamination")
                    # Skip failed batches completely instead of adding dummy data
                    # This prevents feature/label length mismatch and data contamination
                    continue

        # Prepare return dictionary
        result = {}
        
        if extract_logits and all_logits:
            result['logits'] = torch.cat(all_logits, dim=0)
            
        if extract_features and all_features:
            result['features'] = torch.cat(all_features, dim=0).numpy()
        else:
            result['features'] = None
            
        if extract_individual_features and all_individual_features:
            # Concatenate all batches for each modality
            num_modalities = len(all_individual_features[0])
            individual_features_result = []
            
            for modality_idx in range(num_modalities):
                modality_features = []
                for batch_features in all_individual_features:
                    modality_features.append(batch_features[modality_idx])
                
                # Concatenate all batches for this modality
                concatenated = torch.cat(modality_features, dim=0)  # (total_samples, feature_dim)
                individual_features_result.append(concatenated)
            
            result['individual_features'] = individual_features_result
        else:
            result['individual_features'] = None
            
        if all_labels:
            result['labels'] = torch.cat(all_labels, dim=0).numpy()
        else:
            result['labels'] = None
            
        log_msg = f"✅ Extracted data - Logits: {result['logits'].shape if 'logits' in result else 'None'}, "
        log_msg += f"Features: {result['features'].shape if result['features'] is not None else 'None'}, "
        log_msg += f"Labels: {result['labels'].shape if result['labels'] is not None else 'None'}"
        
        if extract_individual_features and result['individual_features'] is not None:
            log_msg += f", Individual: {len(result['individual_features'])} modalities"
        
        logging.info(log_msg)
        
        return result
    
    # Legacy wrapper functions for backward compatibility
    def _extract_logits_batch(self, loader):
        """Legacy function - extracts only logits"""
        result = self._extract_data_batch(loader, extract_features=False, extract_logits=True)
        return result.get('logits', torch.empty(0))
    
    def _extract_features_batch(self, loader):
        """Legacy function - extracts only features and labels"""
        result = self._extract_data_batch(loader, extract_features=True, extract_logits=False)
        return result.get('features'), result.get('labels')
  