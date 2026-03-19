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

# 🎯 UnifiedOODDetector - MSP/Energy/MaxLogit 통합
from ood import UnifiedOODDetector

# 🔍 LTS methods
from ood import LTSFusionDetector

# 🔥 Feature-based OOD methods
# from ood.methods.mahalanobis import MahalanobisDetector

# 📊 Metrics
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
    
    def _compute_total_loss(self, outputs, targets):
        """
        Main Loss와 Auxiliary Loss를 유연하게 결합하여 총 손실 계산
        
        Args:
            outputs: 네트워크 forward 결과 딕셔너리
            targets: 정답 레이블
            
        Returns:
            dict: {
                'total_loss': 최종 손실,
                'main_loss': 주 손실,
                'auxiliary_loss': 보조 손실 (있는 경우),
                'aux_weight': 보조 손실 가중치,
                'has_auxiliary': 보조 손실 사용 여부
            }
        """
        # 1. Main Loss 계산 (항상 존재)
        main_loss = F.cross_entropy(outputs["logits"], targets)
        
        # 2. Fusion 모듈이 자체 compute_total_loss() 메서드를 가지고 있는지 확인
        fusion_module = None
        if hasattr(self._network, 'fusion') and hasattr(self._network.fusion, 'compute_total_loss'):
            fusion_module = self._network.fusion
        elif hasattr(self._network, 'fusion_network') and hasattr(self._network.fusion_network, 'compute_total_loss'):
            fusion_module = self._network.fusion_network
        
        if fusion_module:
            # Fusion 모듈의 compute_total_loss() 사용 (v2_6의 pretrain phase 체크 포함)
            auxiliary_loss = outputs.get('auxiliary_loss', None)
            total_loss = fusion_module.compute_total_loss(main_loss, auxiliary_loss)
            
            # Auxiliary loss 사용 여부 확인 (total_loss가 main_loss보다 크면 사용 중)
            has_auxiliary = auxiliary_loss is not None and total_loss > main_loss
            aux_weight = outputs.get('aux_loss_weight', 0.0) if has_auxiliary else 0.0
            
            return {
                'total_loss': total_loss,
                'main_loss': main_loss,
                'auxiliary_loss': auxiliary_loss if auxiliary_loss is not None else torch.tensor(0.0, device=main_loss.device),
                'aux_weight': aux_weight,
                'has_auxiliary': has_auxiliary
            }
        else:
            # Fusion 모듈이 없거나 compute_total_loss()가 없으면 기존 방식 사용
            has_auxiliary = 'auxiliary_loss' in outputs and outputs['auxiliary_loss'] is not None
            
            if has_auxiliary:
                auxiliary_loss = outputs['auxiliary_loss']
                aux_weight = outputs.get('aux_loss_weight', 0.5)
                
                # Auxiliary loss가 실제로 0이 아닌 경우에만 결합
                if isinstance(auxiliary_loss, torch.Tensor) and auxiliary_loss.item() > 0:
                    total_loss = main_loss + aux_weight * auxiliary_loss
                else:
                    total_loss = main_loss
                    has_auxiliary = False
                
                return {
                    'total_loss': total_loss,
                    'main_loss': main_loss,
                    'auxiliary_loss': auxiliary_loss,
                    'aux_weight': aux_weight if has_auxiliary else 0.0,
                    'has_auxiliary': has_auxiliary
                }
            else:
                # Auxiliary Loss가 없으면 Main Loss만 사용
                return {
                    'total_loss': main_loss,
                    'main_loss': main_loss,
                    'auxiliary_loss': torch.tensor(0.0, device=main_loss.device),
                    'aux_weight': 0.0,
                    'has_auxiliary': False
                }
    
    def _collect_class_confidences(self, phase):
        """
        Class별 modality confidence를 수집하고 출력
        
        Args:
            phase: 로깅 시점 ("START", "FROZEN", "END", "TEST")
        
        Train 시점 (START, FROZEN, END): train_loader 내의 모든 class
        Test 시점 (TEST): test_loader 내의 모든 class (0 ~ total_classes-1)
        """
        # Fusion 모듈이 auxiliary head를 가지고 있는지 확인
        fusion_module = None
        if hasattr(self._network, 'fusion'):
            fusion_module = self._network.fusion
        elif hasattr(self._network, 'fusion_network'):
            fusion_module = self._network.fusion_network
        
        if fusion_module is None or not hasattr(fusion_module, 'auxiliary_heads'):
            logging.info(f"⚠️  No auxiliary heads found - skipping class-wise confidence logging")
            return
        
        # 시점에 따라 적절한 loader 선택
        if phase == "TEST":
            loader = self.test_loader
            loader_desc = "test_loader"
        else:
            loader = self.train_loader
            loader_desc = "train_loader"
        
        # Class별로 confidence를 저장할 딕셔너리
        class_confidences = {}  # {class_id: {modality: [confidences]}}
        
        # 네트워크를 eval 모드로 전환 (학습 중이어도 inference만 수행)
        was_training = self._network.training
        self._network.eval()
        
        logging.info(f"")
        logging.info(f"{'='*80}")
        logging.info(f"📊 Collecting Class-wise Modality Confidences ({phase})")
        logging.info(f"   Task: {self._cur_task}, Epoch: {fusion_module.current_epoch if hasattr(fusion_module, 'current_epoch') else 'N/A'}")
        logging.info(f"   Data Source: {loader_desc}")
        logging.info(f"{'='*80}")
        
        with torch.no_grad():
            for _, inputs, targets in tqdm(loader, desc=f"Collecting confidences ({phase})", leave=False):
                # 입력을 디바이스로 이동
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                
                # Forward pass
                outputs = self._network(inputs)
                
                # Confidence 추출
                if 'confidences' not in outputs or not outputs['confidences']:
                    continue
                
                confidences_dict = outputs['confidences']
                
                # 각 샘플에 대해 class별로 confidence 저장 (필터링 없이 모든 class)
                for i, target in enumerate(targets):
                    class_id = target.item()
                    
                    if class_id not in class_confidences:
                        class_confidences[class_id] = {mod: [] for mod in self._modality}
                    
                    # 각 모달리티의 confidence 저장
                    for modality in self._modality:
                        if modality in confidences_dict:
                            conf_val = confidences_dict[modality][i].item()
                            class_confidences[class_id][modality].append(conf_val)
        
        # 원래 모드로 복원
        if was_training:
            self._network.train()
        
        # Class별 통계 출력
        if class_confidences:
            collected_classes = sorted(class_confidences.keys())
            num_classes = len(collected_classes)
            class_range_str = f"{collected_classes[0]}~{collected_classes[-1]}" if num_classes > 1 else f"{collected_classes[0]}"
            
            logging.info(f"")
            logging.info(f"📈 Class-wise Modality Confidence Statistics:")
            logging.info(f"   Collected {num_classes} classes: {class_range_str}")
            logging.info(f"{'='*80}")
            
            # Class별로 정렬해서 출력
            for class_id in collected_classes:
                logging.info(f"  Class {class_id}:")
                
                for modality in self._modality:
                    if modality in class_confidences[class_id] and class_confidences[class_id][modality]:
                        confs = np.array(class_confidences[class_id][modality])
                        mean_conf = confs.mean()
                        std_conf = confs.std()
                        min_conf = confs.min()
                        max_conf = confs.max()
                        count = len(confs)
                        
                        logging.info(f"    {modality}: mean={mean_conf:.4f}, std={std_conf:.4f}, "
                                   f"min={min_conf:.4f}, max={max_conf:.4f}, count={count}")
            
            logging.info(f"{'='*80}")
            
            # wandb 로깅
            if self.args.get('use_wandb', False):
                wandb_log = {}
                
                for class_id in class_confidences.keys():
                    for modality in self._modality:
                        if modality in class_confidences[class_id] and class_confidences[class_id][modality]:
                            confs = np.array(class_confidences[class_id][modality])
                            wandb_log[f"ClassConfidence/{phase}_class{class_id}_{modality}_mean"] = confs.mean()
                            wandb_log[f"ClassConfidence/{phase}_class{class_id}_{modality}_std"] = confs.std()
                
                wandb_log[f"ClassConfidence/{phase}_task"] = self._cur_task
                wandb_log[f"ClassConfidence/{phase}_epoch"] = fusion_module.current_epoch if hasattr(fusion_module, 'current_epoch') else -1
                
                wandb.log(wandb_log)
                logging.info(f"✅ Logged class-wise confidences to wandb ({phase})")
        else:
            logging.info(f"⚠️  No confidence data collected")
    
    def _setup_epoch_and_collect_confidence(self, epoch):
        """
        Epoch 설정 및 특정 시점에 class별 confidence 수집
        
        Args:
            epoch: 현재 epoch
            
        Returns:
            tuple: (is_first_epoch, is_frozen_epoch, is_last_epoch)
        """
        # 🔥 Fusion 모듈에 현재 epoch 정보 전달
        fusion_module = None
        if hasattr(self._network, 'fusion'):
            fusion_module = self._network.fusion
        elif hasattr(self._network, 'fusion_network'):
            fusion_module = self._network.fusion_network
        
        if fusion_module is not None and hasattr(fusion_module, 'set_epoch'):
            fusion_module.set_epoch(epoch)
        
        # 🎯 Epoch 시점 판단
        pretrain_epochs = 5
        if fusion_module and hasattr(fusion_module, 'pretrain_epochs'):
            pretrain_epochs = fusion_module.pretrain_epochs
        
        is_first_epoch = (epoch == 0)
        is_frozen_epoch = (epoch == pretrain_epochs)
        is_last_epoch = (epoch == self._epochs - 1)
        
        # 🎯 특정 epoch 시작 시점에 class별 confidence 수집
        if is_first_epoch or is_frozen_epoch or is_last_epoch:
            phase = "START" if is_first_epoch else ("FROZEN" if is_frozen_epoch else "END")
            self._collect_class_confidences(phase)
        
        return is_first_epoch, is_frozen_epoch, is_last_epoch
    
    def _update_fusion_task(self):
        """Update fusion model for new task (if supported)"""
        # Check if fusion model supports task updates (e.g., auxiliary_head_v2)
        fusion_module = None
        if hasattr(self._network, 'fusion') and hasattr(self._network.fusion, 'update_task'):
            fusion_module = self._network.fusion
        elif hasattr(self._network, 'fusion_network') and hasattr(self._network.fusion_network, 'update_task'):
            fusion_module = self._network.fusion_network
        
        if fusion_module:
            # 🔥 Warm-up 상태 디버깅
            if hasattr(fusion_module, 'warmup_epochs') and hasattr(fusion_module, 'current_epoch'):
                logging.info(f"🔥 Fusion Warm-up Status BEFORE update_task:")
                logging.info(f"   Task: {self._cur_task}, Current Epoch: {fusion_module.current_epoch}")
                logging.info(f"   Warm-up Epochs: {fusion_module.warmup_epochs}")
                logging.info(f"   Is Warm-up Phase: {fusion_module._is_warmup_phase() if hasattr(fusion_module, '_is_warmup_phase') else 'N/A'}")
            
            fusion_module.update_task(self._cur_task)
            logging.info(f"🎯 Updated fusion model for Task {self._cur_task}")
            
            # 🔥 Warm-up 상태 디버깅 (업데이트 후)
            if hasattr(fusion_module, 'warmup_epochs') and hasattr(fusion_module, 'current_epoch'):
                logging.info(f"🔥 Fusion Warm-up Status AFTER update_task:")
                logging.info(f"   Task: {self._cur_task}, Current Epoch: {fusion_module.current_epoch}")
                logging.info(f"   Is Warm-up Phase: {fusion_module._is_warmup_phase() if hasattr(fusion_module, '_is_warmup_phase') else 'N/A'}")
        else:
            logging.info(f"⚠️  No fusion module with update_task method found for Task {self._cur_task}")
    
    
    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        
        # Update fusion model for new task (if supported)
        self._update_fusion_task()
        
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
            
            # 🎯 Epoch 설정 및 confidence 수집 (공통 메서드 사용)
            self._setup_epoch_and_collect_confidence(epoch)
                
            if self._partialbn:
                self._network.backbone.freeze_fn('partialbn_statistics')
            if self._freeze:
                self._network.backbone.freeze()

            losses = 0.0
            correct, total = 0, 0
            total_batches = len(train_loader)
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                if self.args["debug_mode"] and i >= 5:
                    break
                
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                
                # 🎯 Forward pass with auxiliary loss support
                outputs = self._network(inputs, targets=targets)
                logits = outputs["logits"]
                
                # 🎯 유연한 총 손실 계산 (auxiliary loss가 있을 때만 결합)
                loss_info = self._compute_total_loss(outputs, targets)
                loss = loss_info['total_loss']
                
                # 🎯 디버깅 정보 출력 (auxiliary loss 사용 시)
                if loss_info['has_auxiliary'] and i == 0:  # 첫 번째 배치에서만
                    main_loss_val = loss_info['main_loss'].item()
                    aux_loss_val = loss_info['auxiliary_loss'].item()
                    aux_weight = loss_info['aux_weight']
                    weighted_aux_loss = aux_weight * aux_loss_val
                    total_loss_val = loss_info['total_loss'].item()
                    
                    # 🔥 Fusion 모듈의 warm-up 상태 확인
                    fusion_module = None
                    if hasattr(self._network, 'fusion'):
                        fusion_module = self._network.fusion
                    elif hasattr(self._network, 'fusion_network'):
                        fusion_module = self._network.fusion_network
                    
                    warmup_info = ""
                    if fusion_module and hasattr(fusion_module, '_is_warmup_phase'):
                        is_warmup = fusion_module._is_warmup_phase()
                        current_epoch = getattr(fusion_module, 'current_epoch', 'N/A')
                        warmup_epochs = getattr(fusion_module, 'warmup_epochs', 'N/A')
                        warmup_info = f" | Warm-up: {is_warmup} (Epoch {current_epoch}/{warmup_epochs})"
                    
                    logging.info(f"📊 Multi-task Learning (Task {self._cur_task}, Epoch {epoch}){warmup_info}:")
                    logging.info(f"   🎯 Loss Scale Analysis:")
                    logging.info(f"      Main Loss: {main_loss_val:.4f}")
                    logging.info(f"      Aux Loss (raw): {aux_loss_val:.4f}")
                    logging.info(f"      Aux Weight (λ): {aux_weight}")
                    logging.info(f"      Aux Loss (weighted): {weighted_aux_loss:.4f}")
                    logging.info(f"      Total Loss: {total_loss_val:.4f}")
                    logging.info(f"   📈 Contribution Ratio:")
                    logging.info(f"      Main: {main_loss_val/total_loss_val*100:.1f}% ({main_loss_val:.4f}/{total_loss_val:.4f})")
                    logging.info(f"      Aux: {weighted_aux_loss/total_loss_val*100:.1f}% ({weighted_aux_loss:.4f}/{total_loss_val:.4f})")
                    logging.info(f"   🔍 Loss Ratio: Main/Aux = {main_loss_val/aux_loss_val:.2f}:1")

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
        
        # 🎯 Test 시점에 class별 confidence 수집
        self._collect_class_confidences("TEST")
        
        # Standard CL accuracy evaluation
        cl_metrics = {}
        cnn_accy, nme_accy = self.eval_task()
        
        if nme_accy is not None:
            logging.info(f"CL Accuracy - CNN: {cnn_accy['top1']:.2f}%, NME: {nme_accy['top1']:.2f}%")
        else:
            logging.info(f"CL Accuracy - CNN: {cnn_accy['top1']:.2f}%, NME: Not Available")
            
        # Log task metrics to W&B (모든 CL 메트릭을 한 번에 로깅)
        if self.args.get('use_wandb', False):
            cl_metrics.update({"Task/avg_acc": cnn_accy['top1']})
            
            # CL grouped accuracy 추가
            for k, v in cnn_accy['grouped'].items():
                cl_metrics.update({f"Task/[{k}]_acc": v})

            # NME accuracy 추가 (있는 경우)
            if nme_accy is not None:
                cl_metrics.update({"Task/nme_avg_acc": nme_accy['top1']})
                for k, v in nme_accy.get('grouped', {}).items():
                    cl_metrics.update({f"Task/NME_[{k}]_acc": v})
            
            # 🎯 모든 CL 메트릭을 한 번에 로깅 (동일한 step)
            logging.info("📊 Logging all CL metrics to wandb in a single step...")
            logging.info(f"✅ Logged {len(cl_metrics)} CL metrics to wandb")
        
        return {'cnn': cnn_accy, 'nme': nme_accy if nme_accy else {'top1': 0.0, 'grouped': {}}}, cl_metrics

    def evaluate_ood(self):
        """Evaluate only OOD detection performance"""
        if not self.enable_ood:
            logging.info("OOD evaluation disabled (enable_ood=False).")
            return {}, {}, {}
            
        logging.info(f"=== Task {self._cur_task} OOD Evaluation ===")
        logging.info(f"Known classes: 0-{self._classes_seen_so_far-1}")
        logging.info(f"Unknown classes: {self._classes_seen_so_far}-{self.total_classnum-1}")
        
        # Check OOD configuration
        if "ood_methods" not in self.args:
            logging.error("ood_methods not found in configuration file!")
            return {}, {}, {}
                  
        ood_methods = self.args["ood_methods"]
        logging.info(f"OOD Methods: {ood_methods}")
        
        if self.ood_test_loader is None:
            logging.warning("No OOD test data available. Skipping OOD evaluation.")
            return {}, {}, {}
        
        ood_results = {}
        score_distributions = {}
        ood_methods_metrics = {}  # 모든 OOD 메트릭을 저장할 딕셔너리
        
        logging.info("=== OOD Detection Results ===")
                
        # Check if fusion features are needed (only for LTS_Fusion now)
        need_fusion_features_legacy = "LTS_Fusion" in ood_methods
        
        # Extract data in single forward pass
        print("  📊 Processing ID data (logits + features)...")
        if need_fusion_features_legacy:
            print("  🔍 LTS_Fusion detected - also extracting fusion features...")
        
        id_data = self._extract_data_batch(
            self.test_loader, 
            extract_features=True, 
            extract_logits=True,
            extract_individual_features=False
        )
        id_logits = id_data['logits']
        id_features = id_data['features'] 
        id_labels = id_data['labels']
        
        print("  🎯 Processing OOD data (logits + features)...")
        ood_data = self._extract_data_batch(
            self.ood_test_loader, 
            extract_features=True, 
            extract_logits=True,
            extract_individual_features=False
        )
        ood_logits = ood_data['logits']
        ood_features = ood_data['features']
        ood_labels = ood_data['labels']
        
        print(f"✅ Data extracted - ID: logits{id_logits.shape}, features{id_features.shape}")
        print(f"                   OOD: logits{ood_logits.shape}, features{ood_features.shape}")
        
        # Store extracted data for visualization
        self._cached_id_data = {'features': id_features, 'labels': id_labels}
        self._cached_ood_data = {'features': ood_features, 'labels': ood_labels}

        # 🔥 Auxiliary outputs 사전 수집 (UnifiedOODDetector Hybrid 모드용)
        def needs_auxiliary_outputs(method_name):
            """Check if method needs auxiliary outputs (auxiliary_logits, confidences)"""
            return method_name.startswith(('MSP_Hybrid_', 'Energy_Hybrid_', 'MaxLogit_Hybrid_', 'Entropy_Hybrid_', 'ODIN_Hybrid_'))
        
        def needs_fusion_features(method_name):
            """Check if method needs fusion features (LTS, ReAct, Scale, ASH_S)"""
            return method_name.startswith(('LTS_', 'ReAct_', 'Scale_', 'ASH_S_'))
        
        need_auxiliary_outputs = any(needs_auxiliary_outputs(method) for method in ood_methods)
        need_fusion_features = any(needs_fusion_features(method) for method in ood_methods)
        id_auxiliary_outputs = None
        ood_auxiliary_outputs = None
        
        if need_auxiliary_outputs:
            print("  🔥 UnifiedOODDetector Hybrid methods detected - collecting auxiliary outputs...")
            id_auxiliary_outputs = self._collect_outputs(self.test_loader)
            ood_auxiliary_outputs = self._collect_outputs(self.ood_test_loader)
            
            if id_auxiliary_outputs is not None and ood_auxiliary_outputs is not None:
                print(f"  ✅ Auxiliary outputs collected:")
                print(f"     - Main logits: ✅")
                print(f"     - Auxiliary logits: {list(id_auxiliary_outputs.get('auxiliary_logits', {}).keys())}")
                print(f"     - Confidences: {list(id_auxiliary_outputs.get('confidences', {}).keys())}")

                # ─── [α-Diag] task별 α_m 통계 로깅 + wandb 기록 ────────────
                confs = id_auxiliary_outputs.get('confidences', {})
                if confs:
                    modality_names = list(confs.keys())
                    conf_stacked = torch.stack(
                        [confs[m] for m in modality_names], dim=0
                    )  # [M, N]
                    alpha = torch.softmax(conf_stacked, dim=0)  # [M, N]
                    alpha_np = alpha.detach().cpu().numpy()

                    logging.info(f"[α-Diag] Task {self._cur_task} | ID set α_m stats (N={alpha_np.shape[1]}):")
                    for i, mod in enumerate(modality_names):
                        a = alpha_np[i]
                        logging.info(
                            f"  α_{mod}: mean={a.mean():.4f}, std={a.std():.4f}, "
                            f"min={a.min():.4f}, max={a.max():.4f}"
                        )
                    per_sample_std = alpha_np.std(axis=0)  # [N]
                    logging.info(
                        f"  per-sample std: mean={per_sample_std.mean():.4f}, "
                        f"max={per_sample_std.max():.4f}  "
                        f"(uniform→0.0, one-hot→{((2/3)**0.5)/3:.4f})"
                    )

                    if self.args.get('use_wandb', False):
                        log_dict = {}
                        for i, mod in enumerate(modality_names):
                            a = alpha_np[i]
                            log_dict[f"alpha_diag/{mod}_mean"] = float(a.mean())
                            log_dict[f"alpha_diag/{mod}_std"]  = float(a.std())
                        log_dict["alpha_diag/per_sample_std_mean"] = float(per_sample_std.mean())
                        wandb.log(log_dict, step=self._cur_task)
                # ──────────────────────────────────────────────────────────────
            else:
                print(f"  ❌ ERROR: Auxiliary outputs not available!")
                print(f"     Hybrid methods require auxiliary head fusion, but model doesn't provide auxiliary outputs.")
                print(f"     Please check if your fusion model has auxiliary heads enabled.")
                raise ValueError("Hybrid OOD methods require auxiliary outputs, but they are not available from the model.")

        for method_name in tqdm(ood_methods, desc="OOD Methods", position=0):
            try:
                # 🎯 UnifiedOODDetector: 모든 통합 방법론 처리
                if method_name.startswith(('MSP_', 'Energy_', 'MaxLogit_', 'LTS_', 'ReAct_', 'Scale_', 'ASH_S_', 'ODIN_', 'Entropy_')):
                    try:
                        detector = UnifiedOODDetector.from_method_name(self._network, method_name, device=self._device)
                        logging.info(f"  🔧 Created {method_name} detector:")
                        logging.info(f"     - Method: {detector.method}")
                        logging.info(f"     - Mode: {detector.mode}")
                        logging.info(f"     - Base detector: {detector._base_detector.__class__.__name__}")
                    except ValueError as e:
                        logging.warning(f"⚠️  Failed to parse method name '{method_name}': {e}")
                        continue
                
                # 🔍 LTS legacy method (only LTS_Fusion)
                elif method_name == "LTS_Fusion":
                    detector = LTSFusionDetector(self._network, self._device)
                
                # 🎨 Feature transformation methods (ReAct, Scale, ASH-S)
                elif method_name == "ReAct":
                    from ood import ReActDetector
                    detector = ReActDetector(self._network, self._device, threshold=1.0)
                elif method_name == "Scale":
                    from ood import ScaleDetector
                    detector = ScaleDetector(self._network, self._device, percentile=90)
                elif method_name == "ASH_S":
                    from ood import ASHSDetector
                    detector = ASHSDetector(self._network, self._device, percentile=90)
                
                # 🌡️ ODIN (requires input data)
                elif method_name == "ODIN":
                    from ood import ODINDetector
                    detector = ODINDetector(self._network, self._device, temperature=1000.0, magnitude=0.0014)
                
                else:
                    logging.warning(f"⚠️  Unknown OOD method: {method_name}")
                    continue
                
                logging.info(f"Computing {method_name} scores...")
                
                # Special handling for LTS_Fusion method
                if method_name == "LTS_Fusion":
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
                elif needs_fusion_features(method_name):
                    # 🔥 LTS methods (UnifiedOODDetector LTS_Baseline only)
                    logging.info(f"  🔥 Computing {method_name} with fusion features...")
                    
                    # Check if this is a UnifiedOODDetector (LTS_Baseline)
                    if isinstance(detector, UnifiedOODDetector):
                        # LTS_Baseline: only needs logits + fusion features
                        id_features_tensor = torch.from_numpy(id_features).to(self._device)
                        ood_features_tensor = torch.from_numpy(ood_features).to(self._device)
                        
                        id_outputs = {'logits': id_logits, 'fusion_features': id_features_tensor}
                        ood_outputs = {'logits': ood_logits, 'fusion_features': ood_features_tensor}
                        
                        logging.info(f"     - Mode: {detector.mode}")
                        logging.info(f"     - Logits: {id_logits.shape}")
                        logging.info(f"     - Fusion features: {id_features.shape}")
                        
                        id_scores = detector.compute_scores_from_outputs(id_outputs)
                        ood_scores = detector.compute_scores_from_outputs(ood_outputs)
                    else:
                        # Legacy LTS_Fusion detector (already handled above)
                        logging.warning(f"  ⚠️  {method_name} already handled - skipping")
                        continue
                    
                    logging.info(f"  ✅ Scores computed: ID={len(id_scores)}, OOD={len(ood_scores)}")
                    logging.info(f"     - ID scores: mean={id_scores.mean():.6f}, std={id_scores.std():.6f}, min={id_scores.min():.4f}, max={id_scores.max():.4f}")
                    logging.info(f"     - OOD scores: mean={ood_scores.mean():.6f}, std={ood_scores.std():.6f}, min={ood_scores.min():.4f}, max={ood_scores.max():.4f}")
                
                # 🌡️ ODIN (requires raw input data)
                elif method_name == "ODIN":
                    logging.info(f"  🌡️ ODIN processing (with input perturbation):")
                    logging.info(f"    - Temperature: {detector.temperature}")
                    logging.info(f"    - Magnitude: {detector.magnitude}")
                    
                    # ODIN requires batch-wise processing with raw inputs
                    id_scores_list = []
                    ood_scores_list = []
                    
                    # Process ID data
                    for _, inputs, targets in tqdm(self.test_loader, desc="ODIN ID", leave=False):
                        # Move inputs to device
                        if isinstance(inputs, dict):
                            for m in self._modality:
                                inputs[m] = inputs[m].to(self._device)
                            # For multi-modal, use first modality (RGB)
                            main_input = inputs[self._modality[0]]
                        else:
                            main_input = inputs.to(self._device)
                        
                        # Compute ODIN scores
                        scores = detector.odin_score(main_input)
                        id_scores_list.append(scores)
                    
                    # Process OOD data
                    for _, inputs, targets in tqdm(self.ood_test_loader, desc="ODIN OOD", leave=False):
                        # Move inputs to device
                        if isinstance(inputs, dict):
                            for m in self._modality:
                                inputs[m] = inputs[m].to(self._device)
                            # For multi-modal, use first modality (RGB)
                            main_input = inputs[self._modality[0]]
                        else:
                            main_input = inputs.to(self._device)
                        
                        # Compute ODIN scores
                        scores = detector.odin_score(main_input)
                        ood_scores_list.append(scores)
                    
                    # Concatenate all scores
                    id_scores = np.concatenate(id_scores_list, axis=0)
                    ood_scores = np.concatenate(ood_scores_list, axis=0)
                    
                    logging.info(f"  ✅ ODIN scores computed: ID={len(id_scores)}, OOD={len(ood_scores)}")
                    logging.info(f"     - ID scores: mean={id_scores.mean():.6f}, std={id_scores.std():.6f}")
                    logging.info(f"     - OOD scores: mean={ood_scores.mean():.6f}, std={ood_scores.std():.6f}")
                
                # 🎨 Feature transformation methods (ReAct, Scale, ASH-S)
                elif method_name in ["ReAct", "Scale", "ASH_S"]:
                    if id_features is None or ood_features is None:
                        logging.error(f"{method_name} requires features, but they were not extracted!")
                        continue
                    
                    logging.info(f"  🎨 {method_name} processing:")
                    logging.info(f"    - ID features: {id_features.shape}")
                    logging.info(f"    - OOD features: {ood_features.shape}")
                    
                    # Convert numpy features to tensors
                    id_features_tensor = torch.from_numpy(id_features).to(self._device)
                    ood_features_tensor = torch.from_numpy(ood_features).to(self._device)
                    
                    # Apply transformation
                    if method_name == "ReAct":
                        id_transformed = detector.react(id_features_tensor)
                        ood_transformed = detector.react(ood_features_tensor)
                        logging.info(f"    - ReAct threshold: {detector.threshold}")
                    elif method_name == "Scale":
                        id_transformed = detector.scale(id_features_tensor)
                        ood_transformed = detector.scale(ood_features_tensor)
                        logging.info(f"    - Scale percentile: {detector.percentile}")
                    elif method_name == "ASH_S":
                        id_transformed = detector.ash_s(id_features_tensor)
                        ood_transformed = detector.ash_s(ood_features_tensor)
                        logging.info(f"    - ASH-S percentile: {detector.percentile}")
                    
                    # Pass through FC layer to get logits
                    with torch.no_grad():
                        if hasattr(self._network, 'fc'):
                            id_transformed_logits = self._network.fc(id_transformed)
                            ood_transformed_logits = self._network.fc(ood_transformed)
                        elif hasattr(self._network, 'classifier'):
                            id_transformed_logits = self._network.classifier(id_transformed)
                            ood_transformed_logits = self._network.classifier(ood_transformed)
                        else:
                            logging.error(f"  ❌ Network has no 'fc' or 'classifier' layer!")
                        continue
                    
                    # Compute Energy scores
                    id_scores = torch.logsumexp(id_transformed_logits, dim=1).cpu().numpy()
                    ood_scores = torch.logsumexp(ood_transformed_logits, dim=1).cpu().numpy()
                    
                    logging.info(f"  ✅ {method_name} scores computed: ID={len(id_scores)}, OOD={len(ood_scores)}")
                    logging.info(f"     - ID scores: mean={id_scores.mean():.6f}, std={id_scores.std():.6f}")
                    logging.info(f"     - OOD scores: mean={ood_scores.mean():.6f}, std={ood_scores.std():.6f}")
                
                elif needs_auxiliary_outputs(method_name):
                    # 🔥 UnifiedOODDetector Hybrid modes (non-LTS) - use pre-collected auxiliary outputs
                    if id_auxiliary_outputs is None or ood_auxiliary_outputs is None:
                        logging.error(f"  ❌ {method_name} requires auxiliary outputs but they are not available!")
                        logging.error(f"     This should not happen - auxiliary outputs should have been collected.")
                        raise ValueError(f"Auxiliary outputs required for {method_name} but not available")
                    
                    logging.info(f"  📊 Computing {method_name} with auxiliary outputs...")
                    logging.info(f"     - Main logits: {id_auxiliary_outputs['logits'].shape}")
                    logging.info(f"     - Auxiliary logits: {list(id_auxiliary_outputs['auxiliary_logits'].keys())}")
                    if 'confidences' in id_auxiliary_outputs:
                        logging.info(f"     - Confidences: {list(id_auxiliary_outputs['confidences'].keys())}")
                    
                    try:
                        id_scores = detector.compute_scores_from_outputs(id_auxiliary_outputs)
                        ood_scores = detector.compute_scores_from_outputs(ood_auxiliary_outputs)
                        logging.info(f"  ✅ Scores computed: ID={len(id_scores)}, OOD={len(ood_scores)}")
                        logging.info(f"     - ID scores: mean={id_scores.mean():.6f}, std={id_scores.std():.6f}, min={id_scores.min():.4f}, max={id_scores.max():.4f}")
                        logging.info(f"     - OOD scores: mean={ood_scores.mean():.6f}, std={ood_scores.std():.6f}, min={ood_scores.min():.4f}, max={ood_scores.max():.4f}")
                    except ValueError as e:
                        logging.error(f"  ❌ Failed to compute scores for {method_name}: {e}")
                        raise
                
                else:
                    # Regular methods using only logits (Baseline, Feature-based)
                    if isinstance(detector, UnifiedOODDetector):
                        # UnifiedOODDetector: use compute_scores_from_outputs with logits
                        logging.info(f"  📊 Computing {method_name} (logit-level UnifiedOODDetector)...")
                        id_outputs = {'logits': id_logits}
                        ood_outputs = {'logits': ood_logits}
                        id_scores = detector.compute_scores_from_outputs(id_outputs)
                        ood_scores = detector.compute_scores_from_outputs(ood_outputs)
                    else:
                        # Legacy detectors: use compute_scores_from_cached_logits
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
                    
                    logging.info(f"{method_name}: AUROC={metrics['auroc']:.4f}%, AUPR_ID={metrics['aupr_id']:.4f}%, AUPR_OOD={metrics['aupr_ood']:.4f}%, FPR95={metrics['fpr95']:.4f}%")
                    logging.info(f"  CM@FPR95: TP={cm_fpr95['tp']} FP={cm_fpr95['fp']} TN={cm_fpr95['tn']} FN={cm_fpr95['fn']} | TPR={cm_fpr95['tpr']:.3f} FPR={cm_fpr95['fpr']:.3f}")
                    logging.info(f"  CM@YoudenJ: TP={cm_youden['tp']} FP={cm_youden['fp']} TN={cm_youden['tn']} FN={cm_youden['fn']} | YoudenJ={cm_youden['youdenJ']:.3f}")
                    
                    # 🔄 모든 OOD 메트릭을 하나의 딕셔너리에 수집 (개별 wandb.log 호출 대신)
                    if self.args.get('use_wandb', False):
                        ood_methods_metrics.update({
                            # Core metrics
                            f"Task/{method_name}_auroc": metrics['auroc'],
                            f"Task/{method_name}_aupr_id": metrics['aupr_id'],
                            f"Task/{method_name}_aupr_ood": metrics['aupr_ood'],
                            f"Task/{method_name}_fpr95": metrics['fpr95'],
                            
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
                        })
                else:
                    logging.error(f"{method_name}: Error - {metrics['error']}")
                    
            except Exception as e:
                logging.error(f"{method_name} evaluation failed: {e}")
                ood_results[method_name] = {'error': str(e), 'method': method_name}
        
        # Store results and data for visualization
        self.latest_ood_results = ood_results
        self._visualization_data = {
            'id_features': id_features,
            'id_labels': id_labels,
            'ood_features': ood_features,
            'score_distributions': score_distributions
        }
        
        return ood_results, score_distributions, ood_methods_metrics

    def _compute_energy_stats_from_memory(self, data_manager):
        """
        [Step 1.6 수정] Memory buffer 데이터 기반으로 modality별 energy mean/std 계산.

        수정 이유:
          (1) Train set 데이터는 현재 task에 편향되어 있어 energy 분포 추정에 부적합할 수 있음.
          (2) memory buffer는 prev/current task 데이터가 균형적으로 포함되어 있어
              energy 분포 추정에 더 적합함.
          (3) inference_mode_evaluation에서 memory가 구성되어 있지 않으므로
              build_rehearsal_memory 호출하여 memory를 구성함.
          
        Test: (1) memory 없음 → E_uniform fallback 사용 (stats 미주입)
              (2) memory 있음 → memory 기반 energy stats 사용
        """
        fusion = getattr(self._network, 'fusion', None)
        if fusion is None:
            fusion = getattr(self._network, 'fusion_network', None)
        if fusion is None or not hasattr(fusion, 'auxiliary_heads'):
            logging.info("[EnergyStats] No auxiliary heads – skipping energy stats computation")
            return

        memory = self._get_memory()  # None if empty
        if memory is None:
            logging.info("[EnergyStats] No memory buffer (Task 0 or empty) – E_uniform fallback will be used")
            return  # fusion._energy_stats = {} → _compute_confidence()가 E_uniform fallback 사용

        data_mem, targets_mem = memory
        logging.info(f"[EnergyStats] Computing energy stats from memory buffer ({len(data_mem)} samples) ...")

        mem_dataset = data_manager.get_dataset(
            [], source="train", mode="test",
            appendent=(data_mem, targets_mem)
        )
        mem_loader = DataLoader(
            mem_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

        self._network.eval()
        energy_accum = {mod: [] for mod in self._modality}

        # stats 계산 forward pass 중에는 [α-Diag] 로깅 플래그를 비활성화
        # → 플래그가 소진되지 않아 실제 평가 시 norm=zscore 로그가 정상 출력됨
        for mod_name in ['RGB', 'Gyro', 'Acce']:
            setattr(fusion, f'_energy_logged_{mod_name}', True)  # suppress during stats computation

        with torch.no_grad():
            for i, (_, inputs, _) in enumerate(
                tqdm(mem_loader, desc="EnergyStats(memory) forward", leave=False)
            ):
                if self.args.get("debug_mode") and i >= 10:
                    break
                if isinstance(inputs, dict):
                    for m in inputs:
                        inputs[m] = inputs[m].to(self._device)
                else:
                    inputs = inputs.to(self._device)

                outputs = self._network(inputs)
                aux_logits = outputs.get('auxiliary_logits', {})
                for mod, logits in aux_logits.items():
                    energy = -torch.logsumexp(logits, dim=1)  # [B]
                    energy_accum[mod].append(energy.cpu())

        energy_stats = {}
        for mod, tensors in energy_accum.items():
            if not tensors:
                continue
            all_e = torch.cat(tensors, dim=0)  # [N]
            e_mean = all_e.mean().item()
            e_std  = all_e.std().item()
            energy_stats[mod] = (e_mean, e_std)
            logging.info(
                f"[EnergyStats] {mod}: mean={e_mean:.4f}, std={e_std:.4f}  "
                f"(N={all_e.numel()})"
            )

        if energy_stats:
            fusion.set_energy_stats(energy_stats)
            logging.info("[EnergyStats] ✅ Memory-based energy stats injected into fusion module")
            # stats 주입 후 플래그 리셋 → 실제 평가 시 norm=zscore 로그 출력
            for mod_name in ['RGB', 'Gyro', 'Acce']:
                setattr(fusion, f'_energy_logged_{mod_name}', False)
        else:
            logging.warning("[EnergyStats] ⚠️  No auxiliary logits found – stats not set")

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

        # ── MoAS energy stats 복원 ────────────────────────────────────────
        if 'energy_stats' in checkpoint:
            fusion = getattr(self._network, 'fusion', None) or getattr(self._network, 'fusion_network', None)
            if fusion is not None and hasattr(fusion, 'set_energy_stats'):
                fusion.set_energy_stats(checkpoint['energy_stats'])
                logging.info(f"[MemoryLoad] ✅ Energy stats restored from checkpoint: {list(checkpoint['energy_stats'].keys())}")
        # ─────────────────────────────────────────────────────────────────

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
        
        # 🎯 Reconstruct class_increments for grouped accuracy calculation in eval mode
        # This is needed because after_task() is not called in eval mode
        known_classes = 0
        for i in range(task_id + 1):
            task_size = data_manager.get_task_size(i)
            self.class_increments.append([known_classes, known_classes + task_size - 1])
            known_classes += task_size
        
        # Set _known_classes to the classes seen before the current task
        # For task_id, this is the sum of all previous tasks
        if task_id > 0:
            self._known_classes = sum(data_manager.get_task_size(i) for i in range(task_id))
        else:
            self._known_classes = 0
        
        # Update network classifier to match checkpoint size BEFORE loading
        self._update_classifier(self._total_classes)
        
        # Load checkpoint with correct classifier size
        checkpoint = self.load_checkpoint(checkpoint_path)
        energy_stats_loaded = 'energy_stats' in checkpoint

        # ─── task별 energy 로깅 플래그 리셋 ────────────────────────────
        fusion = getattr(self._network, 'fusion', None)
        if fusion is not None:
            for mod in ['RGB', 'Gyro', 'Acce']:
                setattr(fusion, f'_energy_logged_{mod}', False)
        # ────────────────────────────────────────────────────────────────

        # Ensure model is on correct device
        self._network = self._network.to(self._device)

        # Setup data loaders for evaluation
        self._setup_data_loaders_with_ood(data_manager)

        if hasattr(self, 'test_loader') and self.test_loader is not None:
            logging.info(f"✅ test_loader properly set with {len(self.test_loader.dataset)} samples")
        else:
            logging.error("❌ test_loader is not set properly!")
            raise AttributeError("test_loader was not set up correctly")

        # ── Energy stats: checkpoint에 있으면 재계산 생략 ──────────────
        if energy_stats_loaded:
            logging.info("[EnergyStats] ✅ Loaded from checkpoint — skipping build_rehearsal_memory")
        else:
            # Fallback: legacy checkpoint (no energy_stats) → rebuild memory and recompute
            logging.info("[EnergyStats] Not in checkpoint — rebuilding memory to compute stats")
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
            logging.info(f"[MemoryBuild] Memory built: {len(self._data_memory)} samples")
            self._compute_energy_stats_from_memory(data_manager)
        # ────────────────────────────────────────────────────────────────
        
        # Perform evaluation
        if self.enable_ood:
            # Perform both CL and OOD evaluation
            cl_results, cl_metrics = self.evaluate_cl()
            ood_results, score_distributions, ood_metrics = self.evaluate_ood()
            self.auto_wandb_log(cl_metrics, ood_metrics, self._cur_task + 1)
            return cl_results, ood_results, score_distributions
        else:
            # Perform only CL evaluation
            cl_results, cl_metrics = self.evaluate_cl()
            self.auto_wandb_log(cl_metrics, {}, self._cur_task + 1)
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
        cl_results, cl_metrics = self.evaluate_cl()
        
        if self.enable_ood:
            ood_results, score_distributions, ood_metrics = self.evaluate_ood()
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
    
    def _collect_outputs(self, loader):
        """
        모델의 전체 outputs를 수집 (auxiliary_logits, confidences, modality_weights 포함)
        
        Args:
            loader: DataLoader
            
        Returns:
            dict: 집계된 outputs {
                'logits': tensor,
                'auxiliary_logits': {modality: tensor},
                'confidences': {modality: tensor},
                'modality_weights': tensor
            }
            또는 None (auxiliary head fusion이 아닌 경우)
        """
        self._network.eval()
        
        all_logits = []
        all_auxiliary_logits = {}
        all_confidences = {}
        all_modality_weights = []
        
        with torch.no_grad():
            for _, inputs, targets in tqdm(loader, desc="Collecting outputs", leave=False):
                if isinstance(inputs, dict):
                    for m in inputs:
                        inputs[m] = inputs[m].to(self._device)
                else:
                    inputs = inputs.to(self._device)
                
                # Forward pass
                outputs = self._network(inputs)
                
                # Check if this is auxiliary head fusion
                if 'auxiliary_logits' not in outputs or not outputs['auxiliary_logits']:
                    return None  # Not auxiliary head fusion
                
                # Collect logits
                all_logits.append(outputs['logits'].cpu())
                
                # Collect auxiliary logits
                for modality, aux_logits in outputs['auxiliary_logits'].items():
                    if modality not in all_auxiliary_logits:
                        all_auxiliary_logits[modality] = []
                    all_auxiliary_logits[modality].append(aux_logits.cpu())
                
                # Collect confidences
                if 'confidences' in outputs and outputs['confidences']:
                    for modality, conf in outputs['confidences'].items():
                        if modality not in all_confidences:
                            all_confidences[modality] = []
                        all_confidences[modality].append(conf.cpu())
                
                # Collect modality weights
                if 'modality_weights' in outputs and outputs['modality_weights'] is not None:
                    all_modality_weights.append(outputs['modality_weights'].cpu())
        
        # Concatenate all batches
        result = {
            'logits': torch.cat(all_logits, dim=0)
        }
        
        # Concatenate auxiliary logits
        if all_auxiliary_logits:
            result['auxiliary_logits'] = {}
            for modality, logits_list in all_auxiliary_logits.items():
                result['auxiliary_logits'][modality] = torch.cat(logits_list, dim=0)
        
        # Concatenate confidences
        if all_confidences:
            result['confidences'] = {}
            for modality, conf_list in all_confidences.items():
                result['confidences'][modality] = torch.cat(conf_list, dim=0)
        
        # Concatenate modality weights
        if all_modality_weights:
            result['modality_weights'] = torch.cat(all_modality_weights, dim=0)
        
        return result
    
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
                    
                    # 2. Features 추출 (after fusion 출력)
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
  
    def auto_wandb_log(self, cl_metrics, ood_metrics, task_id):
        if self.args["use_wandb"]:
            all_metrics = {}
            all_metrics.update(cl_metrics)
            all_metrics.update(ood_metrics)
            all_metrics.update({"Task/Task_ID": task_id})
            
            # 🎯 모든 CL + OOD 메트릭을 한 번에 로깅 (동일한 step)
            logging.info("📊 Logging all CL + OOD metrics to wandb in a single step...")
            wandb.log(all_metrics)
            logging.info(f"✅ Logged {len(cl_metrics)} CL + {len(ood_metrics)} OOD metrics to wandb")