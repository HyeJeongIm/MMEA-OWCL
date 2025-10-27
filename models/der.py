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
from utils.toolkit import tensor2numpy
from models.baseline_tbn import TBNBaseline
from models.baseline_tsn import TSNBaseline

EPSILON = 1e-8


class DER(Replay):
    """
    🎯 Dark Experience Replay (DER)
    
    핵심 아이디어:
    1. 기존 Replay: input + target만 저장하여 재학습
    2. DER: 이전 모델의 예측 logits도 함께 저장 (dark knowledge)
    3. 재학습 시 distillation loss로 이전 지식 보존
    
    메모리 구조:
    - _data_memory: input data
    - _targets_memory: target labels  
    - _logits_memory: 이전 모델의 예측 logits (dark knowledge)
    
    Loss:
    - Current task: CrossEntropy (input, target)
    - Rehearsal: KL Divergence (old_logits, new_logits)
    
    논문: "Dark Experience for General Continual Learning" (arxiv:2004.07211)
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        # 🎯 DER 하이퍼파라미터
        self.der_alpha = args.get("der_alpha", 0.5)  # DER loss weight
        
        # 🎯 Logits 메모리 초기화 (old model predictions)
        self._logits_memory = np.array([])
        
        logging.info(f"🎯 DER initialized with alpha={self.der_alpha}")
    
    def incremental_train(self, data_manager):
        """Override incremental_train to store old predictions"""
        self.total_classnum = data_manager.get_total_classnum()
        
        # 🎯 Save data_manager reference
        self._data_manager = data_manager

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._classes_seen_so_far = self._total_classes
        self.class_increments.append([self._known_classes, self._total_classes - 1])

        self._network.update_fc(self._total_classes)
        logging.info(f"Learning on {self._known_classes}-{self._total_classes}")

        self._setup_data_loaders_with_ood(data_manager)

        # 🎯 DataParallel 설정 (network를 GPU로 이동시킴)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        # 🎯 Store old predictions after network is on GPU
        if self._cur_task > 0 and hasattr(self, '_data_memory') and self._data_memory.size > 0:
            logging.info("🎯 Storing old predictions for DER...")
            self._store_old_predictions()
            self._setup_der_train_loaders(data_manager)

        self._train(self.train_loader, self.test_loader)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        # 🎯 DataParallel 해제 전 network 모듈 가져오기
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def _setup_der_train_loaders(self, data_manager):
        """DER 전용 DataLoader 설정 - old logits를 함께 반환"""
        logging.info(f"Setting up DER train loaders for Task {self._cur_task}")
        
        # 🎯 Get memory with logits
        memory_data = self._get_memory()
        if memory_data is not None and self._cur_task > 0 and hasattr(self, '_logits_memory') and len(self._logits_memory) > 0:
            # Memory에 logits 추가
            appendent = (memory_data[0], memory_data[1], self._logits_memory)
        else:
            appendent = memory_data
        
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=appendent,
        )
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers
        )
    
    def _store_old_predictions(self):
        """메모리의 샘플들에 대해 현재 모델의 예측값을 저장"""
        # 🎯 data_manager를 사용하여 dataset 생성
        # This will properly handle the VideoRecord objects and transformations
        if not hasattr(self, '_data_manager'):
            logging.warning("🎯 No data_manager available, skipping old predictions storage")
            return
        
        memory_dataset = self._data_manager.get_dataset(
            [], source="train", mode="train", 
            appendent=self._get_memory()
        )
        
        memory_loader = DataLoader(
            memory_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers
        )
        
        # 현재 모델로 예측값 추출
        self._network.to(self._device)
        self._network.eval()
        old_logits = []
        
        with torch.no_grad():
            for _, inputs, _ in memory_loader:
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                
                # 🎯 Forward pass (forward는 dict 반환)
                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.forward(inputs)
                else:
                    outputs = self._network.forward(inputs)
                
                # 🎯 Outputs에서 logits 추출
                if isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                    old_logits.append(tensor2numpy(logits))
                else:
                    logging.warning("🎯 No logits found in network output, skipping batch")
                    continue
        
        self._logits_memory = np.concatenate(old_logits)
        logging.info(f"🎯 Stored old predictions for {len(self._logits_memory)} exemplars")
    
    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        """🎯 DER: Dark Knowledge Distillation 포함"""
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        schedulers = scheduler if isinstance(scheduler, (list, tuple)) else [scheduler]

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
            der_losses = 0.0  # 🎯 DER loss tracking
            
            for i, batch in enumerate(train_loader):
                if self.args["debug_mode"] and i >= 5:
                    break

                # 🎯 Handle both regular dataset (3 values) and DER dataset (4 values)
                if len(batch) == 4:
                    _, inputs, targets, old_logits_batch = batch
                else:
                    _, inputs, targets = batch
                    old_logits_batch = None

                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)

                # 🎯 Forward pass
                outputs = self._network(inputs, targets=targets)
                logits = outputs["logits"]
                
                # 🎯 Loss 계산 (Standard + DER)
                loss_info = self._compute_total_loss(outputs, targets)
                main_loss = loss_info['total_loss']
                
                # 🎯 DER Loss 추가 (old logits와의 distillation)
                if old_logits_batch is not None:
                    # Convert old_logits to tensor
                    old_logits = old_logits_batch.float().to(self._device)
                        
                    # 🎯 Filter out rows where ALL elements are -1
                    # valid_mask: True for rows that have at least one value != -1
                    der_loss = self._compute_der_loss(logits, targets, old_logits)
                    
                    total_loss = main_loss + der_loss
                    der_losses += der_loss.item()
                        
                else:
                    total_loss = main_loss
                    der_loss = torch.tensor(0.0, device=self._device)

                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                total_loss.backward()
                if self._clip_gradient is not None:
                    nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)
                for opt in optimizers:
                    opt.step()

                losses += main_loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.numel()

            for sch in schedulers:
                sch.step()

            train_acc = round((correct * 100.0) / max(1, total), 2)
            avg_der_loss = der_losses / len(train_loader) if len(train_loader) > 0 else 0.0

            # wandb 로깅
            if self.args["use_wandb"]:
                wandb.log({
                    "Train/train_loss": losses / len(train_loader),
                    "Train/train_accuracy": train_acc,
                    "Train/der_loss": avg_der_loss,
                })

            info = f"Task {self._cur_task}, Epoch {epoch+1}/{self._epochs} => Loss {losses/len(train_loader):.3f}, Train_accy {train_acc:.2f}, DER_loss {avg_der_loss:.3f}"
            if self.args.get("log_test_acc", False) and epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
                if self.args["use_wandb"]:
                    wandb.log({"Train/test_accuracy": test_acc})

            prog_bar.set_description(info)

        logging.info(info)
        
    def _compute_der_loss(self, logits, targets, old_logits):
        """
        🎯 DER Loss 계산
        
        Args:
            logits: 현재 모델의 logits [batch_size, num_classes]
            targets: targets [batch_size]
            old_logits: 이전 모델의 logits [batch_size, num_classes]
        """
        valid_mask = (old_logits != -1).any(dim=1)
        
        # 🎯 NaN 방지: valid한 샘플이 없으면 0 반환
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self._device)
        
        return self.der_alpha * F.mse_loss(logits[valid_mask], old_logits[valid_mask])


class TBN_DER(DER):
    """DER model for TBN backbone"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TBNBaseline(args)


class TSN_DER(DER):
    """DER model for TSN backbone"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TSNBaseline(args)


# =====================================================
# DER++ (Enhanced DER with additional penalties)
# =====================================================

class DERpp(DER):
    """
    🎯 Dark Experience Replay++ (DER++)
    
    DER에서 다음을 추가:
    1. DER의 원래 loss
    2. Old targets에 대한 additional penalty (consistent predictions)
    
    Loss:
    L = L_current(x, y) + α × L_logits(z', h(x_buffer)) + β × L_cons(z', y_buffer)
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        # DER++는 beta를 사용
        self.der_beta = args.get("der_beta", 0.5)    # DER loss balance parameter
        
        logging.info(f"🎯 DER++ initialized with alpha={self.der_alpha}, beta={self.der_beta}")
        
    def _compute_der_loss(self, logits, targets, old_logits):
        """
        🎯 DER++ Loss 계산
        
        Args:
            logits: 현재 모델의 logits [batch_size, num_classes]
            targets: targets [batch_size]
            old_logits: 이전 모델의 logits [batch_size, num_classes]
        """
        # DER의 기본 loss
        der_loss = super()._compute_der_loss(logits, targets, old_logits)
        
        # 🎯 DER++: Consistency loss 추가
        valid_mask = (old_logits != -1).any(dim=1)
        
        if valid_mask.sum() == 0:
            return der_loss
        
        der_cons_loss = F.cross_entropy(logits[valid_mask], targets[valid_mask])
        
        return der_loss + self.der_beta * der_cons_loss


class TBN_DERpp(DERpp):
    """DER++ model for TBN backbone"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TBNBaseline(args)


class TSN_DERpp(DERpp):
    """DER++ model for TSN backbone"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TSNBaseline(args)
