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
    
    def _store_old_predictions(self):
        """메모리의 샘플들에 대해 현재 모델의 예측값을 저장"""
        # 🎯 data_manager를 사용하여 dataset 생성
        prev_start, prev_end = self.class_increments[-2]  # range inclusive
        targets_mem = self._targets_memory
        mask_idx = np.where((targets_mem >= prev_start) & (targets_mem <= prev_end))[0]
        mask_idx_not = np.where((targets_mem < prev_start) | (targets_mem > prev_end))[0]

        selected_data = self._data_memory[mask_idx]
        selected_targets = self._targets_memory[mask_idx]

        memory_dataset = self._data_manager.get_dataset(
            [],
            source="train",
            mode="train",
            appendent=(selected_data, selected_targets)
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
                    np_logits = tensor2numpy(logits).astype(np.float32)
                    if np_logits.ndim == 1:
                        np_logits = np_logits.reshape(1, -1)
                    old_logits.append(np_logits)
                else:
                    logging.warning("🎯 No logits found in network output, skipping batch")
                    continue

        old_logits_np = np.concatenate(old_logits, axis=0).astype(np.float32)
        
        if len(self._logits_memory) == 0:
            self._logits_memory = old_logits_np
        else:
            logits_memory = np.full((len(targets_mem), self._total_classes), 0, dtype=np.float32) # 0 means no logits for MSE Loss
            logits_memory[mask_idx_not, :self._known_classes] = self._logits_memory
            logits_memory[mask_idx, :self._total_classes] = old_logits_np
            self._logits_memory = logits_memory
        
        logging.info(f"🎯 Stored old predictions for {len(self._logits_memory)} exemplars")
        assert len(self._logits_memory) == len(self._data_memory)
    
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
    
    def _reduce_exemplar_reservoir(self, data_manager, m):
        """
        🎯 기존 클래스에 대해 Reservoir Sampling 방식으로 축소
        - 각 클래스별로 m개를 무작위 선택하여 축소
        """
        logging.info(f"🎯 Reducing exemplars with Reservoir Sampling...({m} per class)")
        
        # 기존 메모리 백업
        dummy_data = copy.deepcopy(self._data_memory)
        dummy_targets = copy.deepcopy(self._targets_memory)
        dummy_logits = copy.deepcopy(self._logits_memory)
        
        # 메모리 초기화
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self._logits_memory = np.array([])
        
        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            class_data = dummy_data[mask]
            class_logits = dummy_logits[mask]
            
            m_current = min(m, len(class_data))
            if m_current > 0:
                # Reservoir sampling: uniform random sampling
                indices = np.random.choice(len(class_data), size=m_current, replace=False)
                dd = class_data[indices]
                dt = dummy_targets[mask][indices]
                dl = class_logits[indices]
                
                self._data_memory = (
                    np.concatenate((self._data_memory, dd))
                    if len(self._data_memory) != 0
                    else dd
                )
                self._targets_memory = (
                    np.concatenate((self._targets_memory, dt))
                    if len(self._targets_memory) != 0
                    else dt
                )
                self._logits_memory = (
                    np.concatenate((self._logits_memory, dl))
                    if len(self._logits_memory) != 0
                    else dl
                )
                
                logging.info(f"  ✅ Class {class_idx}: {m_current} exemplars selected")
                
    def _construct_exemplar_reservoir(self, data_manager, m):
        """
        🎯 Reservoir Sampling 방식으로 exemplar 구성
        - 새 클래스에 대해 m개의 샘플을 무작위로 선택 (uniform 확률)
        - 클래스 평균 계산에는 사용하지 않음 (NME 없음)
        """
        logging.info(f"🎯 Constructing exemplars with Reservoir Sampling...({m} per class)")
        
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            
            # 🎯 Reservoir Sampling 구현
            m = min(m, data.shape[0])
            
            # 간단한 random sampling (uniform 확률)
            # 실제 reservoir sampling을 하려면 전체 데이터를 스트림으로 처리해야 함
            indices = np.random.choice(data.shape[0], size=m, replace=False)
            selected_exemplars = data[indices]
            exemplar_targets = np.full(m, class_idx)
            
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )
            
            # Reservoir sampling은 class mean을 별도로 계산하지 않음
            # (NME classifier 사용 안 함)
            logging.info(f"  ✅ Class {class_idx}: {m} exemplars selected via Reservoir Sampling")
    
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
        valid_mask = (old_logits != -1).all(dim=1)
        
        # 🎯 NaN 방지: valid한 샘플이 없으면 0 반환
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self._device)
        
        mask = (old_logits != 0).float()  # old_logits==0 → 0, 나머지 → 1
        masked_logits = logits * mask
        
        return self.der_alpha * F.mse_loss(
            masked_logits[valid_mask],
            old_logits[valid_mask]
        )
        # Temperature-scaled softmax for knowledge distillation to avoid logit scale mismatch or stabilize the learning process
        # KL Divergence Loss (Knowledge Distillation)
        # der_loss = F.kl_div(
        #     F.log_softmax(masked_logits[valid_mask] / self.der_temp, dim=1),
        #     F.softmax(old_logits[valid_mask] / self.der_temp, dim=1),
        #     reduction='batchmean'
        # )



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
