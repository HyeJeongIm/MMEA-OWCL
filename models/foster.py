import logging
import copy
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models.replay import Replay
from utils.toolkit import tensor2numpy, count_parameters
from models.baseline_tbn import TBNBaseline
from models.baseline_tsn import TSNBaseline

EPSILON = 1e-8


class FOSTER(Replay):
    """
    🎯 FOSTER for Multi-modal Continual Learning
    
    핵심 아이디어:
    1. Knowledge Distillation: 이전 태스크의 지식을 새 모델로 전달
    2. Feature Extractor Loss: 현재 feature → 이전 classifier
    3. Class-balanced Learning: Imbalanced class distribution 처리
    4. Weight Alignment: Old/New classes의 classifier bias 조정
    5. Feature Compression: Student network로 dark knowledge 압축
    
    Replay를 상속받아 multimodal 처리는 완벽하게 동작!
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        # 🎯 FOSTER 하이퍼파라미터
        self.lambda_okd = args.get("lambda_okd", 1.0)  # Knowledge distillation weight
        self.lambda_fe = args.get("lambda_fe", 1.0)  # Feature extractor loss weight
        self.beta1 = args.get("beta1", 0.96)  # Class reweighting parameter (boosting)
        self.beta2 = args.get("beta2", 0.97)  # Class reweighting parameter (compression)
        self.is_teacher_wa = args.get("is_teacher_wa", False)  # Weight alignment (teacher)
        self.is_student_wa = args.get("is_student_wa", False)  # Weight alignment (student)
        self.wa_value = args.get("wa_value", 1.0)  # Weight alignment scale
        self.use_compression = args.get("use_compression", True)  # Enable feature compression
        
        # 🎯 FOSTER 특화 변수
        self._old_network = None  # Store old model for distillation
        self._old_fc = None  # Store old classifier for fe_logits
        self._snet = None  # Student network for feature compression
        self.per_cls_weights = None  # Class-balanced weights
        
        logging.info(f"🎯 FOSTER initialized with lambda_okd={self.lambda_okd}, lambda_fe={self.lambda_fe}, beta1={self.beta1}, beta2={self.beta2}, wa={self.is_teacher_wa}, compression={self.use_compression}")
    
    def after_task(self):
        """Override to save old classifier and apply weight alignment"""
        # Call parent's after_task
        super().after_task()
        
        # 🎯 Save old classifier for fe_logits (FOSTER 특징)
        if self._network is not None and hasattr(self._network, 'fc'):
            self._old_fc = copy.deepcopy(self._network.fc)
            self._old_fc.eval()
            for p in self._old_fc.parameters():
                p.requires_grad = False
            logging.info(f"✅ Saved old classifier with {self._total_classes} classes")
        
        # 🎯 Weight Alignment (FOSTER 특징)
        if self._cur_task > 0 and self.is_teacher_wa:
            self._weight_align(
                self._known_classes,
                self._total_classes - self._known_classes,
                self.wa_value
            )
            logging.info(f"✅ Applied weight alignment with factor {self.wa_value}")
    
    def incremental_train(self, data_manager):
        """Override to add old network storage and compression"""
        self.total_classnum = data_manager.get_total_classnum()
        self._cur_task += 1
        
        # 🎯 Use student network from previous task (FOSTER 특징)
        if self._cur_task > 1 and self._snet is not None:
            logging.info("🔄 Replacing teacher with student network from previous task")
            self._network = self._snet
            self._snet = None
        
        # 🎯 Store old network for distillation (Task 1+)
        if self._cur_task > 0:
            self._old_network = self._network.copy()
            self._old_network = self._old_network.to(self._device)  # 🔥 GPU로 이동!
            self._old_network.freeze()
            logging.info("✅ Stored old network for knowledge distillation")
        
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._classes_seen_so_far = self._total_classes
        self.class_increments.append([self._known_classes, self._total_classes - 1])

        self._network.update_fc(self._total_classes)
        logging.info(f"Learning on {self._known_classes}-{self._total_classes}")

        # Count parameters
        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info("Trainable params: {}".format(count_parameters(self._network, True)))

        self._setup_data_loaders_with_ood(data_manager)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def _train(self, train_loader, test_loader):
        """Override to add feature compression phase"""
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
            # 🎯 Feature Boosting: Train with fe_logits + KD
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            
            # 🎯 Feature Compression: Train student network (optional)
            if self.use_compression and self._cur_task > 0:
                logging.info("🎓 Starting Feature Compression phase...")
                self._feature_compression(train_loader, test_loader)
    
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """Initial training for Task 0 (same as Replay)"""
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        schedulers = scheduler if isinstance(scheduler, (list, tuple)) else [scheduler]
        
        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            
            # 🎯 Epoch 설정 및 confidence 수집
            self._setup_epoch_and_collect_confidence(epoch)
            
            if self._partialbn:
                self._network.backbone.freeze_fn('partialbn_statistics')
            if self._freeze:
                self._network.backbone.freeze_fn('bn_statistics')
            
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                if self.args.get("debug_mode", False) and i >= 5:
                    break
                
                # 🎯 Multi-modal input handling
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                
                # 🎯 Forward pass
                outputs = self._network(inputs, targets=targets)
                logits = outputs["logits"]
                
                # 🎯 Compute total loss
                loss_info = self._compute_total_loss(outputs, targets)
                loss = loss_info['total_loss']
                
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                loss.backward()
                if self._clip_gradient is not None:
                    nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)
                for opt in optimizers:
                    opt.step()
                    
                losses += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.numel()
                
            for sch in schedulers:
                sch.step()
                
            train_acc = round((correct * 100.0) / max(1, total), 2)
            
            # Log to wandb
            if self.args.get("use_wandb", False):
                wandb.log({
                    "Train/train_loss": losses / len(train_loader),
                    "Train/train_accuracy": train_acc,
                })
            
            info = f"Task {self._cur_task}, Epoch {epoch+1}/{self._epochs} => Loss {losses/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            if self.args.get("log_test_acc", False) and epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
                if self.args.get("use_wandb", False):
                    wandb.log({"Train/test_accuracy": test_acc})
            
            prog_bar.set_description(info)
        logging.info(info)
    
    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        """🎯 Training with FOSTER-style knowledge distillation (Task 1+)"""
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        schedulers = scheduler if isinstance(scheduler, (list, tuple)) else [scheduler]
        
        # 🎯 Compute class-balanced weights (FOSTER 특징)
        if self._cur_task > 0:
            self.per_cls_weights = self._compute_class_weights()
            logging.info(f"🎯 Class-balanced weights computed with beta1={self.beta1}")

        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            
            # 🎯 Epoch 설정 및 confidence 수집
            self._setup_epoch_and_collect_confidence(epoch)

            if self._partialbn:
                self._network.backbone.freeze_fn("partialbn_statistics")
            if self._freeze:
                self._network.backbone.freeze_fn("bn_statistics")

            losses, correct, total = 0.0, 0, 0
            losses_clf, losses_fe, losses_kd = 0.0, 0.0, 0.0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                if self.args.get("debug_mode", False) and i >= 5:
                    break

                # 🎯 Multi-modal input handling
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)

                # 🎯 Forward pass (현재 네트워크)
                outputs = self._network(inputs, targets=targets)
                logits = outputs["logits"]
                
                # 🎯 Classification loss with class reweighting (FOSTER 특징)
                if self.per_cls_weights is not None:
                    weights = self.per_cls_weights[targets]
                    loss_clf = F.cross_entropy(logits, targets, reduction='none')
                    loss_clf = (loss_clf / weights).mean()
                else:
                    loss_info = self._compute_total_loss(outputs, targets)
                    loss_clf = loss_info['total_loss']
                
                # 🎯 Feature Extractor Loss: 현재 feature → 이전 classifier (FOSTER 특징!)
                loss_fe = torch.tensor(0.0, device=self._device)
                if self._old_fc is not None:
                    # 🎯 old_fc는 old classes만 예측 가능 → old class samples만 필터링
                    old_class_mask = targets < self._known_classes
                    if old_class_mask.any():
                        with torch.no_grad():
                            features = self._network.extract_vector(inputs)
                        fe_logits = self._old_fc(features)
                        if isinstance(fe_logits, dict):
                            fe_logits = fe_logits["logits"]
                        # Old class samples만 loss 계산
                        loss_fe = self.lambda_fe * F.cross_entropy(
                            fe_logits[old_class_mask], 
                            targets[old_class_mask]
                        )
                        losses_fe += loss_fe.item()
                
                # 🎯 Knowledge Distillation loss (FOSTER 특징!)
                loss_kd = torch.tensor(0.0, device=self._device)
                if self._old_network is not None:
                    with torch.no_grad():
                        old_outputs = self._old_network(inputs, targets=targets)
                        old_logits = old_outputs["logits"]
                    # KD loss only on old classes
                    loss_kd = self.lambda_okd * _KD_loss(
                        logits[:, : self._known_classes], 
                        old_logits[:, : self._known_classes], 
                        self.args.get("T", 2)
                    )
                    losses_kd += loss_kd.item()
                
                # 🎯 Total loss = Classification + Feature Extractor + Knowledge Distillation
                total_loss = loss_clf + loss_fe + loss_kd

                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                total_loss.backward()
                if self._clip_gradient is not None:
                    nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)
                for opt in optimizers:
                    opt.step()

                losses += total_loss.item()
                losses_clf += loss_clf.item()
                preds = torch.argmax(logits, dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.numel()

            for sch in schedulers:
                sch.step()

            train_acc = round((correct * 100.0) / max(1, total), 2)
            avg_clf_loss = losses_clf / len(train_loader) if len(train_loader) > 0 else 0.0
            avg_fe_loss = losses_fe / len(train_loader) if len(train_loader) > 0 else 0.0
            avg_kd_loss = losses_kd / len(train_loader) if len(train_loader) > 0 else 0.0

            # wandb 로깅
            if self.args.get("use_wandb", False):
                wandb.log({
                    "Train/train_loss": losses / len(train_loader),
                    "Train/train_accuracy": train_acc,
                    "Train/loss_clf": avg_clf_loss,
                    "Train/loss_fe": avg_fe_loss,
                    "Train/loss_kd": avg_kd_loss,
                })

            info = f"Task {self._cur_task}, Epoch {epoch+1}/{self._epochs} => Loss {losses/len(train_loader):.3f}, Train_accy {train_acc:.2f}, CLF {avg_clf_loss:.3f}, FE {avg_fe_loss:.3f}, KD {avg_kd_loss:.3f}"
            if self.args.get("log_test_acc", False) and epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
                if self.args.get("use_wandb", False):
                    wandb.log({"Train/test_accuracy": test_acc})

            prog_bar.set_description(info)

        logging.info(info)
    
    def _compute_class_weights(self):
        """🎯 Compute class-balanced weights using effective number (FOSTER 특징)"""
        cls_num_list = []
        
        # Old classes: memory samples
        if hasattr(self, '_targets_memory') and len(self._targets_memory) > 0:
            for class_idx in range(self._known_classes):
                count = np.sum(self._targets_memory == class_idx)
                cls_num_list.append(max(count, 1))
        else:
            cls_num_list = [self.samples_per_class] * self._known_classes
        
        # New classes: training samples (estimate)
        new_class_samples = self._memory_size // (self._total_classes - self._known_classes) if (self._total_classes - self._known_classes) > 0 else self.samples_per_class
        for _ in range(self._known_classes, self._total_classes):
            cls_num_list.append(new_class_samples)
        
        # Compute effective number
        effective_num = 1.0 - np.power(self.beta1, cls_num_list)
        per_cls_weights = (1.0 - self.beta1) / np.array(effective_num)
        
        # Normalize
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        
        logging.info(f"📊 Class weights: {per_cls_weights[:5]}... (showing first 5)")
        
        return torch.FloatTensor(per_cls_weights).to(self._device)
    
    def _weight_align(self, old_class_num, new_class_num, gamma=1.0):
        """🎯 Weight Alignment: Balance old and new class classifier weights (FOSTER 특징)"""
        # Get classifier weights
        if hasattr(self._network, 'fc'):
            if hasattr(self._network.fc, 'fc_action'):
                fc_weights = self._network.fc.fc_action.weight.data
            elif hasattr(self._network.fc, 'weight'):
                fc_weights = self._network.fc.weight.data
            else:
                logging.warning("⚠️ Cannot find classifier weights for weight alignment")
                return
        else:
            logging.warning("⚠️ Network has no 'fc' attribute for weight alignment")
            return
        
        # Compute old and new weight norms
        old_weights = fc_weights[:old_class_num]
        new_weights = fc_weights[old_class_num:old_class_num + new_class_num]
        
        old_norm = torch.norm(old_weights, p=2, dim=1).mean()
        new_norm = torch.norm(new_weights, p=2, dim=1).mean()
        
        if new_norm > 1e-8:
            # 🎯 원본 FOSTER의 exponential scaling
            scaling_factor = (old_norm / new_norm) * (gamma ** (old_class_num / new_class_num))
            fc_weights[old_class_num:old_class_num + new_class_num] *= scaling_factor
            
            logging.info(f"📐 Weight alignment: old_norm={old_norm:.4f}, new_norm={new_norm:.4f}, scaling={scaling_factor:.4f}")
        else:
            logging.warning("⚠️ New weights have near-zero norm, skipping alignment")
    
    def _feature_compression(self, train_loader, test_loader):
        """🎯 Feature Compression: Train student network with BKD loss (FOSTER 특징)"""
        # 🎯 Create student network (same architecture as teacher)
        if isinstance(self._network, TBNBaseline):
            self._snet = TBNBaseline(self.args)
        elif isinstance(self._network, TSNBaseline):
            self._snet = TSNBaseline(self.args)
        else:
            logging.error(f"❌ Unsupported network type for compression: {type(self._network)}")
            return
        
        self._snet.update_fc(self._total_classes)
        self._snet.to(self._device)
        
        # 🎯 Recompute class weights with beta2
        cls_num_list = []
        if hasattr(self, '_targets_memory') and len(self._targets_memory) > 0:
            for class_idx in range(self._known_classes):
                count = np.sum(self._targets_memory == class_idx)
                cls_num_list.append(max(count, 1))
        else:
            cls_num_list = [self.samples_per_class] * self._known_classes
        
        new_class_samples = self._memory_size // (self._total_classes - self._known_classes) if (self._total_classes - self._known_classes) > 0 else self.samples_per_class
        for _ in range(self._known_classes, self._total_classes):
            cls_num_list.append(new_class_samples)
        
        effective_num = 1.0 - np.power(self.beta2, cls_num_list)
        per_cls_weights = (1.0 - self.beta2) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)
        
        logging.info(f"📊 Compression class weights (beta2={self.beta2}): {per_cls_weights[:5]}...")
        
        # 🎯 Setup optimizer and scheduler
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._snet.parameters()),
            lr=self.args.get("lr", 0.1),
            momentum=0.9,
            weight_decay=self.args.get("weight_decay", 5e-4)
        )
        
        compression_epochs = self.args.get("compression_epochs", self._epochs)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=compression_epochs
        )
        
        # 🎯 Freeze teacher network
        self._network.eval()
        for p in self._network.parameters():
            p.requires_grad = False
        
        # 🎯 Train student network
        prog_bar = tqdm(range(compression_epochs))
        for _, epoch in enumerate(prog_bar):
            self._snet.train()
            
            # 🎯 Epoch 설정
            self._setup_epoch_and_collect_confidence(epoch)
            
            if self._partialbn:
                self._snet.backbone.freeze_fn("partialbn_statistics")
            if self._freeze:
                self._snet.backbone.freeze_fn("bn_statistics")
            
            losses = 0.0
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                if self.args.get("debug_mode", False) and i >= 5:
                    break
                
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                
                # Student forward
                student_outputs = self._snet(inputs, targets=targets)
                student_logits = student_outputs["logits"]
                
                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_outputs = self._network(inputs, targets=targets)
                    teacher_logits = teacher_outputs["logits"]
                
                # BKD loss
                loss_bkd = self._BKD_loss(
                    student_logits,
                    teacher_logits,
                    self.args.get("T", 2)
                )
                
                optimizer.zero_grad()
                loss_bkd.backward()
                if self._clip_gradient is not None:
                    nn.utils.clip_grad_norm_(self._snet.parameters(), self._clip_gradient)
                optimizer.step()
                
                losses += loss_bkd.item()
                preds = torch.argmax(student_logits, dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.numel()
            
            scheduler.step()
            
            train_acc = round((correct * 100.0) / max(1, total), 2)
            
            if self.args.get("use_wandb", False):
                wandb.log({
                    "Compression/train_loss": losses / len(train_loader),
                    "Compression/train_accuracy": train_acc,
                })
            
            info = f"🎓 SNet: Task {self._cur_task}, Epoch {epoch+1}/{compression_epochs} => Loss {losses/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            prog_bar.set_description(info)
        
        logging.info(info)
        
        # 🎯 Weight Alignment for student (optional)
        if self.is_student_wa:
            self._weight_align_snet(
                self._known_classes,
                self._total_classes - self._known_classes,
                self.wa_value
            )
        
        # 🎯 Unfreeze teacher network
        for p in self._network.parameters():
            p.requires_grad = True
    
    def _BKD_loss(self, student_logits, teacher_logits, T):
        """🎯 Balanced Knowledge Distillation Loss"""
        student_log_softmax = torch.log_softmax(student_logits / T, dim=1)
        teacher_softmax = torch.softmax(teacher_logits / T, dim=1)
        
        # Apply per-class weights
        if self.per_cls_weights is not None:
            teacher_softmax = teacher_softmax * self.per_cls_weights
            teacher_softmax = teacher_softmax / teacher_softmax.sum(1, keepdim=True)
        
        return -1 * torch.mul(teacher_softmax, student_log_softmax).sum() / student_logits.shape[0]
    
    def _weight_align_snet(self, old_class_num, new_class_num, gamma=1.0):
        """🎯 Weight Alignment for Student Network"""
        if not hasattr(self._snet, 'fc'):
            return
            
        if hasattr(self._snet.fc, 'fc_action'):
            fc_weights = self._snet.fc.fc_action.weight.data
        elif hasattr(self._snet.fc, 'weight'):
            fc_weights = self._snet.fc.weight.data
        else:
            return
        
        old_weights = fc_weights[:old_class_num]
        new_weights = fc_weights[old_class_num:old_class_num + new_class_num]
        
        old_norm = torch.norm(old_weights, p=2, dim=1).mean()
        new_norm = torch.norm(new_weights, p=2, dim=1).mean()
        
        if new_norm > 1e-8:
            scaling_factor = (old_norm / new_norm) * (gamma ** (old_class_num / new_class_num))
            fc_weights[old_class_num:old_class_num + new_class_num] *= scaling_factor
            logging.info(f"📐 Student weight alignment: old_norm={old_norm:.4f}, new_norm={new_norm:.4f}, scaling={scaling_factor:.4f}")


def _KD_loss(pred, soft, T):
    """Knowledge Distillation loss"""
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


class TBN_FOSTER(FOSTER):
    """FOSTER with TBN multi-modal backbone"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TBNBaseline(args)


class TSN_FOSTER(FOSTER):
    """FOSTER with TSN multi-modal backbone"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TSNBaseline(args)
