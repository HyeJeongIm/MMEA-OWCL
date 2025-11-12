import logging
import copy
import os

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


class MMEADER(Replay):
    """
    🎯 Multimodal Dark Experience Replay (MMEDER)
    
    핵심 아이디어:
    1. 기존 Replay: input + target만 저장하여 재학습
    2. DER: 이전 모델의 예측 logits도 함께 저장 (dark knowledge)
    3. 재학습 시 distillation loss로 이전 지식 보존
    
    메모리 구조:
    - _data_memory: input data
    - _targets_memory: target labels  
    - _logits_memory: 이전 모델의 예측 logits (dark knowledge)
    
    Loss:
    - Current task: CrossEntropy (input, target) for each modality
    - Rehearsal: KL Divergence (old_logits, new_logits)
    
    논문: "Dark Experience for General Continual Learning" (arxiv:2004.07211)
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        # 🎯 DER 하이퍼파라미터
        self.mmeader_alpha = args.get("mmeader_alpha", 0.5)  # DER loss weight
        self.mmeader_temp = args.get("mmeader_temp", 4.0)     # Temperature for distillation
        
        # 🎯 Auxiliary logits 메모리 (모달리티별 보조 헤드 로짓을 concat하여 저장)
        self._auxiliary_logits_memory = np.array([])
        
        logging.info(f"🎯 MMEADER initialized with alpha={self.mmeader_alpha}, temperature={self.mmeader_temp}")
    
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
            logging.info("🎯 Storing old auxiliary predictions for MMEADER...")
            self._store_old_predictions()
            self._setup_der_train_loaders(data_manager)

        self._train(self.train_loader, self.test_loader)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        # 🎯 DataParallel 해제 전 network 모듈 가져오기
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def _store_old_predictions(self):
        """메모리 샘플들에 대한 현재 모델의 'auxiliary_logits'를 저장 (모달리티 기준 concat)"""
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
        old_aux_concat_list = []
        
        with torch.no_grad():
            for _, inputs, _ in memory_loader:
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                
                # 🎯 Forward pass (forward는 dict 반환)
                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.forward(inputs)
                else:
                    outputs = self._network.forward(inputs)
                
                # 🎯 Outputs에서 auxiliary_logits 추출 후 모달리티 순서대로 concat
                if isinstance(outputs, dict) and 'auxiliary_logits' in outputs and isinstance(outputs['auxiliary_logits'], dict):
                    aux_dict = outputs['auxiliary_logits']
                    np_slices = []
                    for m in self._modality:
                        if m in aux_dict and aux_dict[m] is not None:
                            aux_logits = aux_dict[m]  # Tensor [B, C]
                            np_aux = tensor2numpy(aux_logits).astype(np.float32)
                            if np_aux.ndim == 1:
                                np_aux = np_aux.reshape(1, -1)
                            np_slices.append(np_aux)
                        else:
                            # 해당 모달리티가 없으면 0으로 채우기 (크기 추정 필요 → 다른 모달리티의 C로 대체 불가)
                            # 안전하게는 현재까지 np_slices가 비어있으면 스킵
                            pass
                    if len(np_slices) > 0:
                        old_aux_concat_list.append(np.concatenate(np_slices, axis=1))
                else:
                    logging.warning("🎯 No logits found in network output, skipping batch")
                    continue

        old_aux_concat = np.concatenate(old_aux_concat_list, axis=0).astype(np.float32)
        num_modalities = len(self._modality)
        classes_now = old_aux_concat.shape[1] // max(1, num_modalities)

        if len(self._auxiliary_logits_memory) == 0:
            self._auxiliary_logits_memory = old_aux_concat
        else:
            # 새 메모리 텐서를 0으로 만들고, 이전 logits는 각 모달리티 슬라이스의 known_classes까지만 복사
            full_width = num_modalities * self._total_classes
            logits_memory = np.full((len(targets_mem), full_width), 0, dtype=np.float32)

            prev_width = self._auxiliary_logits_memory.shape[1]
            prev_classes = prev_width // max(1, num_modalities)

            for mi in range(num_modalities):
                # 이전 메모리 복사: known_classes까지
                prev_start = mi * prev_classes
                prev_end = prev_start + self._known_classes
                new_start = mi * self._total_classes
                new_end_known = new_start + self._known_classes
                logits_memory[mask_idx_not, new_start:new_end_known] = self._auxiliary_logits_memory[:, prev_start:prev_end]

                # 현재 예측 복사: total_classes까지
                cur_start = mi * classes_now
                cur_end = cur_start + self._total_classes
                logits_memory[mask_idx, new_start:cur_end] = old_aux_concat[:, cur_start:cur_end]

            self._auxiliary_logits_memory = logits_memory

        logging.info(f"🎯 Stored old auxiliary predictions for {len(self._auxiliary_logits_memory)} exemplars")
        assert len(self._auxiliary_logits_memory) == len(self._data_memory)


    def _setup_der_train_loaders(self, data_manager):
        """DER 전용 DataLoader 설정 - old logits를 함께 반환"""
        logging.info(f"Setting up DER train loaders for Task {self._cur_task}")
        
        # 🎯 Get memory with auxiliary logits
        memory_data = self._get_memory()
        if memory_data is not None and self._cur_task > 0 and hasattr(self, '_auxiliary_logits_memory') and len(self._auxiliary_logits_memory) > 0:
            # Memory에 logits 추가
            appendent = (memory_data[0], memory_data[1], self._auxiliary_logits_memory)
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
        dummy_logits = copy.deepcopy(self._auxiliary_logits_memory)
        
        # 메모리 초기화
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self._auxiliary_logits_memory = np.array([])
        
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
                self._auxiliary_logits_memory = (
                    np.concatenate((self._auxiliary_logits_memory, dl))
                    if len(self._auxiliary_logits_memory) != 0
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
    
    def _reduce_exemplar(self, data_manager, m):
        """
        🎯 MMEADER 버전: 기존 클래스 exemplar 축소 (class means 계산 + auxiliary logits 처리)
        Replay의 _reduce_exemplar를 오버라이드하여 auxiliary logits도 함께 처리
        """
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        dummy_logits = copy.deepcopy(self._auxiliary_logits_memory) if len(self._auxiliary_logits_memory) > 0 else np.array([])
        
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self._auxiliary_logits_memory = np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            
            # Auxiliary logits도 함께 축소
            if len(dummy_logits) > 0:
                dl = dummy_logits[mask][:m]
            else:
                dl = np.array([])
            
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
            
            # Auxiliary logits 메모리 업데이트
            if len(dl) > 0:
                self._auxiliary_logits_memory = (
                    np.concatenate((self._auxiliary_logits_memory, dl))
                    if len(self._auxiliary_logits_memory) != 0
                    else dl
                )

            # Exemplar mean 계산 (Replay와 동일)
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean
            
    def _construct_exemplar(self, data_manager, m):
        """
        🎯 MMEADER 버전: 새 클래스 exemplar 구성 (class mean 기반 선택 + auxiliary logits 초기화)
        Replay의 _construct_exemplar를 오버라이드하여 auxiliary logits도 함께 처리
        """
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select exemplars (Replay와 동일한 방식)
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]

            m = min(m, vectors.shape[0])
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    data[i]
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    vectors[i]
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
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
            
            # 🎯 새 exemplar에 대한 auxiliary logits는 아직 없으므로 빈 배열로 초기화
            # 나중에 _store_old_predictions에서 채워질 예정
            num_modalities = len(self._modality)
            aux_logits_width = num_modalities * self._total_classes
            new_aux_logits = np.full((m, aux_logits_width), 0, dtype=np.float32)  # 0으로 초기화 (나중에 채워짐)
            
            self._auxiliary_logits_memory = (
                np.concatenate((self._auxiliary_logits_memory, new_aux_logits))
                if len(self._auxiliary_logits_memory) != 0
                else new_aux_logits
            )

            # Exemplar mean 계산 (Replay와 동일)
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            
            self._class_means[class_idx, :] = mean
    
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
            der_losses = 0.0  # 🎯 Auxiliary DER loss tracking
            
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
                
                # 🎯 Loss 계산 (Standard + DER)
                loss_info = self._compute_total_loss(outputs, targets)
                main_loss = loss_info['total_loss']
                
                # 🎯 DER Loss 추가 (old logits와의 distillation)
                if old_logits_batch is not None:
                    # Convert old_logits to tensor
                    old_aux_logits = old_logits_batch.float().to(self._device)

                    # 현재 auxiliary logits dict를 모달리티 순서로 concat
                    aux_dict = outputs.get('auxiliary_logits', {}) if isinstance(outputs, dict) else {}
                    curr_slices = []
                    for m in self._modality:
                        if isinstance(aux_dict, dict) and m in aux_dict and aux_dict[m] is not None:
                            curr_slices.append(aux_dict[m])
                    if len(curr_slices) > 0:
                        current_aux_concat = torch.cat(curr_slices, dim=1)
                    else:
                        current_aux_concat = None

                    der_loss = self._compute_aux_der_loss(current_aux_concat, old_aux_logits)
                    
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

                losses += total_loss.item()
                preds = torch.argmax(outputs["logits"], dim=1)
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
                    "Train/aux_der_loss": avg_der_loss,
                })

            info = f"Task {self._cur_task}, Epoch {epoch+1}/{self._epochs} => Loss {losses/len(train_loader):.3f}, Train_accy {train_acc:.2f}, Aux_DER_loss {avg_der_loss:.3f}"
            if self.args.get("log_test_acc", False) and epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
                if self.args["use_wandb"]:
                    wandb.log({"Train/test_accuracy": test_acc})

            prog_bar.set_description(info)

        logging.info(info)
        
    def _compute_aux_der_loss(self, current_aux_concat, old_aux_concat):
        """
        🎯 Auxiliary DER Loss 계산 (모달리티별 auxiliary logits concat 기준)
        
        Args:
            current_aux_concat: Tensor [B, M*C]
            old_aux_concat: Tensor [B, M*C]
        """
        if current_aux_concat is None:
            return torch.tensor(0.0, device=self._device)

        valid_mask = (old_aux_concat != -1).all(dim=1)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self._device)

        mask = (old_aux_concat != 0).float()
        old_aux_concat = old_aux_concat / self.mmeader_temp
        current_aux_concat = current_aux_concat / self.mmeader_temp
        masked_current = current_aux_concat * mask
        return self.mmeader_alpha * F.mse_loss(
            masked_current[valid_mask],
            old_aux_concat[valid_mask]
        )
        
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
            for batch in tqdm(loader, desc=f"Collecting confidences ({phase})", leave=False):
                # 입력을 디바이스로 이동
                if len(batch) == 4:
                    _, inputs, targets, old_logits_batch = batch
                else:
                    _, inputs, targets = batch
                    old_logits_batch = None

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
    
    def save_checkpoint(self, weights_dir, filename):
        """
        🎯 MMEADER 모델의 체크포인트 저장
        파라미터 정보(alpha, temp, aux_loss_weight)를 경로에 반영하여 저장
        """
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        
        # iCaRL: Save class means for NME evaluation
        if hasattr(self, '_class_means'):
            save_dict['class_means'] = self._class_means
            logging.info(f"💾 Saved class means for {len(self._class_means)} classes")
        
        # 🎯 MMEADER 파라미터 정보를 경로에 반영
        alpha = self.mmeader_alpha
        temp = self.mmeader_temp
        aux_weight = self.args.get("aux_loss_weight", 0.5)
        
        # 파라미터 정보를 포함한 서브디렉토리 생성
        param_subdir = f"alpha{alpha}_temp{temp}_aux{aux_weight}"
        weights_dir = os.path.join(weights_dir, param_subdir)
        os.makedirs(weights_dir, exist_ok=True)
        logging.info(f"🎯 MMEADER: Saving to parameter-specific directory: {param_subdir}")
        
        torch.save(save_dict, "{}/{}_{}.pkl".format(weights_dir, filename, self._cur_task))

class TBN_MMEADER(MMEADER):
    """DER model for TBN backbone"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TBNBaseline(args)


class TSN_MMEADER(MMEADER):
    """DER model for TSN backbone"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TSNBaseline(args)

 
