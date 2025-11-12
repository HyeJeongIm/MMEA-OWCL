import logging
import copy
import os
from collections import defaultdict

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
        
        # 🎯 Auxiliary logits 메모리 (모달리티별 보조 헤드 로짓을 dictionary로 저장)
        self._auxiliary_logits_memory = defaultdict(lambda: np.array([]))
        
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
            # self._store_old_predictions()
            self._setup_der_train_loaders(data_manager)

        self._train(self.train_loader, self.test_loader)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        # 🎯 DataParallel 해제 전 network 모듈 가져오기
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def _store_old_predictions(self):
        """메모리 샘플들에 대한 현재 모델의 'auxiliary_logits'를 저장 (모달리티별 dict 형태)"""
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
        old_aux_dict = defaultdict(list)  # {modality: [logits]}
        
        with torch.no_grad():
            for _, inputs, _ in memory_loader:
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                
                # 🎯 Forward pass (forward는 dict 반환)
                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.forward(inputs)
                else:
                    outputs = self._network.forward(inputs)
                
                # 🎯 Outputs에서 auxiliary_logits 추출 후 모달리티별로 분리하여 저장
                if isinstance(outputs, dict) and 'auxiliary_logits' in outputs and isinstance(outputs['auxiliary_logits'], dict):
                    aux_dict = outputs['auxiliary_logits']
                    for m in self._modality:
                        if m in aux_dict and aux_dict[m] is not None:
                            aux_logits = aux_dict[m]  # Tensor [B, C]
                            np_aux = tensor2numpy(aux_logits).astype(np.float32)
                            if np_aux.ndim == 1:
                                np_aux = np_aux.reshape(1, -1)
                            old_aux_dict[m].append(np_aux)
                else:
                    logging.warning("🎯 No logits found in network output, skipping batch")
                    continue

        # 모달리티별로 concat
        old_aux_logits = {}
        for m in self._modality:
            if m in old_aux_dict and len(old_aux_dict[m]) > 0:
                old_aux_logits[m] = np.concatenate(old_aux_dict[m], axis=0).astype(np.float32)
            else:
                old_aux_logits[m] = np.array([])

        # 기존 메모리와 병합 (모달리티별로 처리)
        if isinstance(self._auxiliary_logits_memory, dict):
            # dict 형태인 경우
            for m in self._modality:
                if m in old_aux_logits and len(old_aux_logits[m]) > 0:
                    # 새 메모리 텐서 생성: [num_samples, total_classes]
                    num_samples = len(targets_mem)
                    logits_memory = np.zeros((num_samples, self._total_classes), dtype=np.float32)
                    
                    # 이전 메모리에서 known_classes까지 복사
                    if m in self._auxiliary_logits_memory and len(self._auxiliary_logits_memory[m]) > 0:
                        prev_logits = self._auxiliary_logits_memory[m]
                        prev_classes = prev_logits.shape[1]
                        known_classes_to_copy = min(self._known_classes, prev_classes)
                        logits_memory[mask_idx_not, :known_classes_to_copy] = prev_logits[mask_idx_not, :known_classes_to_copy]
                    
                    # 현재 예측 복사: total_classes까지
                    cur_logits = old_aux_logits[m]
                    cur_classes = cur_logits.shape[1]
                    classes_to_copy = min(self._total_classes, cur_classes)
                    logits_memory[mask_idx, :classes_to_copy] = cur_logits[:, :classes_to_copy]
                    
                    self._auxiliary_logits_memory[m] = logits_memory
        else:
            # 기존 numpy array 형태 (하위 호환성 - 사용되지 않음)
            logging.warning("⚠️  _auxiliary_logits_memory is not dict, converting...")
            self._auxiliary_logits_memory = old_aux_logits

        num_stored = len(targets_mem)
        logging.info(f"🎯 Stored old auxiliary predictions for {num_stored} exemplars")
        assert num_stored == len(self._data_memory)

    def _setup_der_train_loaders(self, data_manager):
        """DER 전용 DataLoader 설정 - old logits를 함께 반환
        모달리티별 dict를 data_manager에 직접 전달 (data_manager에서 변환 처리)
        또는 numpy array로 변환하여 전달 (하위 호환성)
        """
        logging.info(f"Setting up DER train loaders for Task {self._cur_task}")
        
        # 🎯 Get memory with auxiliary logits
        memory_data = self._get_memory()
        if memory_data is not None and self._cur_task > 0 and hasattr(self, '_auxiliary_logits_memory'):
            # dict 형태의 auxiliary_logits_memory를 data_manager에 직접 전달
            # data_manager가 dict를 받아서 모달리티 순서대로 concat하여 처리
            if isinstance(self._auxiliary_logits_memory, dict):
                # dict 형태: data_manager가 처리하도록 직접 전달
                appendent = (memory_data[0], memory_data[1], self._auxiliary_logits_memory)
            elif len(self._auxiliary_logits_memory) > 0:
                # 기존 numpy array 형태 (하위 호환성)
                appendent = (memory_data[0], memory_data[1], self._auxiliary_logits_memory)
            else:
                appendent = memory_data
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
        모달리티별로 dict 형태로 처리
        """
        logging.info(f"🎯 Reducing exemplars with Reservoir Sampling...({m} per class)")
        
        # 기존 메모리 백업
        dummy_data = copy.deepcopy(self._data_memory)
        dummy_targets = copy.deepcopy(self._targets_memory)
        dummy_logits = copy.deepcopy(self._auxiliary_logits_memory)
        
        # 메모리 초기화
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self._auxiliary_logits_memory = defaultdict(lambda: np.array([]))
        
        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            class_data = dummy_data[mask]
            
            m_current = min(m, len(class_data))
            if m_current > 0:
                # Reservoir sampling: uniform random sampling
                indices = np.random.choice(len(class_data), size=m_current, replace=False)
                dd = class_data[indices]
                dt = dummy_targets[mask][indices]
                
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
                
                # 모달리티별로 logits 처리
                for mod in self._modality:
                    if isinstance(dummy_logits, dict) and mod in dummy_logits and len(dummy_logits[mod]) > 0:
                        class_logits = dummy_logits[mod][mask]
                        dl = class_logits[indices]
                        if mod in self._auxiliary_logits_memory and len(self._auxiliary_logits_memory[mod]) > 0:
                            self._auxiliary_logits_memory[mod] = np.concatenate(
                                (self._auxiliary_logits_memory[mod], dl), axis=0
                            )
                        else:
                            self._auxiliary_logits_memory[mod] = dl
                
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
    
    def _extract_vectors_and_auxiliary_logits(self, loader):
        """모달리티별로 auxiliary logits를 dict로 반환"""
        self._network.eval()
        vectors, targets = [], []
        auxiliary_logits_dict = defaultdict(list)  # {modality: [logits]}
        
        for _, _inputs, _targets in loader:
            for m in self._modality:
                _inputs[m] = _inputs[m].to(self._device)
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _outputs = self._network.module.forward(_inputs)
            else:
                _outputs = self._network.forward(_inputs)
            
            _vectors = tensor2numpy(self._consensus(_outputs['features']))
            
            # 모달리티별로 분리하여 저장
            for m in self._modality:
                _m_auxiliary_logits = tensor2numpy(_outputs['auxiliary_logits'][m])
                auxiliary_logits_dict[m].append(_m_auxiliary_logits)
                
            vectors.append(_vectors)
            targets.append(_targets)
        
        # 모달리티별로 concat
        auxiliary_logits = {}
        for m in self._modality:
            if m in auxiliary_logits_dict:
                auxiliary_logits[m] = np.concatenate(auxiliary_logits_dict[m], axis=0)
            else:
                auxiliary_logits[m] = np.array([])
        
        return np.concatenate(vectors), np.concatenate(targets), auxiliary_logits
    
    def _reduce_exemplar(self, data_manager, m):
        """
        🎯 MMEADER 버전: 기존 클래스 exemplar 축소 (class means 계산 + auxiliary logits 처리)
        Replay의 _reduce_exemplar를 오버라이드하여 auxiliary logits도 함께 처리
        모달리티별로 dict 형태로 처리
        """
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data = copy.deepcopy(self._data_memory)
        dummy_targets =  copy.deepcopy(self._targets_memory)
        dummy_logits = copy.deepcopy(self._auxiliary_logits_memory)
        
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self._auxiliary_logits_memory = defaultdict(lambda: np.array([]))

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            
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
            
            # 모달리티별로 logits 처리
            for mod in self._modality:
                if isinstance(dummy_logits, dict) and mod in dummy_logits and len(dummy_logits[mod]) > 0:
                    dl = dummy_logits[mod][mask][:m]
                    if mod in self._auxiliary_logits_memory and len(self._auxiliary_logits_memory[mod]) > 0:
                        self._auxiliary_logits_memory[mod] = np.concatenate(
                            (self._auxiliary_logits_memory[mod], dl), axis=0
                        )
                    else:
                        self._auxiliary_logits_memory[mod] = dl

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
        모달리티별로 dict 형태로 저장하여 클래스 인덱스 일관성 유지
        """
        logging.info("Constructing exemplars...({} per classes)".format(m))
        
        # 기존 메모리가 있으면 모달리티별로 next task dimension에 맞게 padding
        if self._known_classes > 0:
            next_logits_dim = self._total_classes + self.args['increment']
            for mod in self._modality:
                if mod in self._auxiliary_logits_memory and len(self._auxiliary_logits_memory[mod]) > 0:
                    cur_logits_dim = self._auxiliary_logits_memory[mod].shape[1]
                    if cur_logits_dim < next_logits_dim:
                        self._auxiliary_logits_memory[mod] = np.pad(
                            self._auxiliary_logits_memory[mod], 
                            ((0, 0), (0, next_logits_dim - cur_logits_dim)), 
                            mode='constant', constant_values=0
                        )
        
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
            vectors, _, auxiliary_logits_dict = self._extract_vectors_and_auxiliary_logits(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select exemplars (Replay와 동일한 방식)
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            exemplar_auxiliary_logits_dict = defaultdict(list)  # {modality: [logits]}

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
                
                # 모달리티별로 logits 저장
                for mod in self._modality:
                    if mod in auxiliary_logits_dict:
                        exemplar_auxiliary_logits_dict[mod].append(auxiliary_logits_dict[mod][i])

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection
                # 모달리티별로 삭제
                for mod in self._modality:
                    if mod in auxiliary_logits_dict:
                        auxiliary_logits_dict[mod] = np.delete(auxiliary_logits_dict[mod], i, axis=0)

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            
            # 모달리티별로 exemplar logits 배열 생성 및 padding
            next_logits_dim = self._total_classes + self.args['increment']
            for mod in self._modality:
                if mod in exemplar_auxiliary_logits_dict:
                    exemplar_aux_logits = np.array(exemplar_auxiliary_logits_dict[mod])  # [m, cur_logits_dim]
                    cur_logits_dim = exemplar_aux_logits.shape[1]
                    if cur_logits_dim < next_logits_dim:
                        exemplar_aux_logits = np.pad(
                            exemplar_aux_logits, 
                            ((0, 0), (0, next_logits_dim - cur_logits_dim)),
                            mode='constant', constant_values=0
                        )
                    
                    # 메모리에 추가
                    if mod in self._auxiliary_logits_memory and len(self._auxiliary_logits_memory[mod]) > 0:
                        self._auxiliary_logits_memory[mod] = np.concatenate(
                            (self._auxiliary_logits_memory[mod], exemplar_aux_logits), axis=0
                        )
                    else:
                        self._auxiliary_logits_memory[mod] = exemplar_aux_logits
            
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
            auxiliary_losses, der_losses = 0.0, 0.0  # 🎯 Auxiliary DER loss tracking
            # 🎯 모달리티별 auxiliary logit accuracy 추적
            aux_correct = {m: 0 for m in self._modality}
            aux_total = {m: 0 for m in self._modality}
            
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
                auxiliary_loss = loss_info['auxiliary_loss'] * loss_info['aux_weight']
                
                # 🎯 DER Loss 추가 (old logits와의 distillation)
                # old_logits_batch는 dict 형태로 전달됨 (모달리티별로 분리)
                # 각 모달리티별로 독립적으로 loss를 계산하여 합산
                if old_logits_batch is not None:
                    # old_logits_batch는 dict 형태 또는 numpy array 형태일 수 있음
                    # dict 형태인 경우 모달리티별로 처리
                    if isinstance(old_logits_batch, dict):
                        # dict 형태: 모달리티별로 loss 계산
                        der_loss = self._compute_aux_der_loss_modality_wise(
                            outputs.get('auxiliary_logits', {}), old_logits_batch
                        )
                    else:
                        # numpy array 형태 (하위 호환성 - dict로 변환 시도)
                        # 이 경우는 발생하지 않을 것으로 예상되지만 안전을 위해 처리
                        logging.warning("⚠️  Received numpy array old_logits, converting to dict")
                        old_logits_dict = {}
                        # numpy array를 모달리티별로 분리 (추정)
                        aux_dict = outputs.get('auxiliary_logits', {})
                        if isinstance(aux_dict, dict) and len(aux_dict) > 0:
                            # 각 모달리티의 클래스 수를 추정하여 분리
                            old_logits_np = np.array(old_logits_batch) if not isinstance(old_logits_batch, torch.Tensor) else old_logits_batch.cpu().numpy()
                            total_dim = old_logits_np.shape[1] if old_logits_np.ndim >= 2 else 0
                            start_idx = 0
                            for m in self._modality:
                                if m in aux_dict and aux_dict[m] is not None:
                                    num_classes = aux_dict[m].shape[1]
                                    old_logits_dict[m] = old_logits_np[:, start_idx:start_idx + num_classes]
                                    start_idx += num_classes
                        der_loss = self._compute_aux_der_loss_modality_wise(
                            outputs.get('auxiliary_logits', {}), old_logits_dict
                        )
                    
                    total_loss = main_loss + der_loss
                    der_losses += der_loss.item()
                    auxiliary_losses += auxiliary_loss.item()
                        
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
                # 🎯 Main logits accuracy
                preds = torch.argmax(outputs["logits"], dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.numel()
                
                # 🎯 모달리티별 auxiliary logits accuracy 계산
                aux_dict = outputs.get('auxiliary_logits', {})
                if isinstance(aux_dict, dict):
                    for m in self._modality:
                        if m in aux_dict and aux_dict[m] is not None:
                            aux_preds = torch.argmax(aux_dict[m], dim=1)
                            aux_correct[m] += aux_preds.eq(targets).sum().item()
                            aux_total[m] += targets.numel()

            for sch in schedulers:
                sch.step()

            train_acc = round((correct * 100.0) / max(1, total), 2)
            
            # 🎯 모달리티별 auxiliary logits accuracy 계산
            aux_acc_dict = {}
            for m in self._modality:
                if aux_total[m] > 0:
                    aux_acc_dict[m] = round((aux_correct[m] * 100.0) / aux_total[m], 2)
                else:
                    aux_acc_dict[m] = 0.0

            # wandb 로깅
            wandb_log_dict = {
                "Train/train_loss": losses / len(train_loader),
                "Train/aux_loss": auxiliary_losses / len(train_loader),
                "Train/aux_der_loss": der_losses / len(train_loader),
                "Train/train_accuracy": train_acc,
            }
            # 모달리티별 auxiliary accuracy 추가
            for m in self._modality:
                wandb_log_dict[f"Train/aux_acc_{m}"] = aux_acc_dict[m]
            
            if self.args["use_wandb"]:
                wandb.log(wandb_log_dict)

            # info 메시지에 모달리티별 auxiliary accuracy 추가 (출력 코드 정리)
            aux_acc_str = ", ".join([f"Aux_{m}_acc {aux_acc_dict[m]:.2f}" for m in self._modality])
            info = (
                f"Task {self._cur_task}, Epoch {epoch+1}/{self._epochs} => "
                f"Loss {losses/len(train_loader):.3f}, "
                f"Aux_loss {auxiliary_losses/len(train_loader):.3f}, "
                f"Aux_DER_loss {der_losses/len(train_loader):.3f}, "
                f"Train_accy {train_acc:.2f}, "
                f"{aux_acc_str}"
            )
            if self.args.get("log_test_acc", False) and epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
                if self.args["use_wandb"]:
                    wandb.log({"Train/test_accuracy": test_acc})

            prog_bar.set_description(info)

        logging.info(info)
        
    def _compute_aux_der_loss_modality_wise(self, current_aux_dict, old_aux_dict):
        """
        🎯 Auxiliary DER Loss 계산 (모달리티별로 독립적으로 계산하여 합산)
        
        Args:
            current_aux_dict: dict {modality: Tensor [B, C]} - 현재 모델의 auxiliary logits (모달리티별)
            old_aux_dict: dict {modality: numpy array [B, C]} - 이전 모델의 auxiliary logits (모달리티별)
            
        Returns:
            total_loss: Tensor - 모든 모달리티의 loss 합산
            
        Note:
            - 각 모달리티의 클래스 인덱스는 독립적으로 유지됨
            - 예: RGB의 클래스 1은 항상 RGB의 인덱스 1에 위치
            - Flow의 클래스 1은 항상 Flow의 인덱스 1에 위치
        """
        total_loss = torch.tensor(0.0, device=self._device)
        
        if not isinstance(current_aux_dict, dict) or not isinstance(old_aux_dict, dict):
            return total_loss
        
        # 모달리티별로 loss 계산
        for m in self._modality:
            if m not in current_aux_dict or current_aux_dict[m] is None:
                continue
            if m not in old_aux_dict or len(old_aux_dict[m]) == 0:
                continue
            
            current_aux = current_aux_dict[m]  # Tensor [B, C]
            old_aux = old_aux_dict[m]  # numpy array [B, C]
            
            # numpy array를 tensor로 변환
            if isinstance(old_aux, np.ndarray):
                old_aux = torch.from_numpy(old_aux).float().to(self._device)
            elif isinstance(old_aux, torch.Tensor):
                old_aux = old_aux.float().to(self._device)
            else:
                continue
            
            # shape 확인
            if current_aux.shape[0] != old_aux.shape[0]:
                continue
            
            # valid mask: -1이 아닌 경우만 유효
            valid_mask = (old_aux != -1).all(dim=1)
            if valid_mask.sum() == 0:
                continue
            
            # mask: 0이 아닌 경우만 loss 계산
            mask = (old_aux != 0).float()
            
            # Temperature scaling
            old_aux_scaled = old_aux / self.mmeader_temp
            current_aux_scaled = current_aux / self.mmeader_temp
            
            # Masked loss
            masked_current = current_aux_scaled * mask
            
            # MSE loss 계산
            modality_loss = F.mse_loss(
                masked_current[valid_mask],
                old_aux_scaled[valid_mask]
            )
            
            total_loss = total_loss + self.mmeader_alpha * modality_loss
        
        return total_loss
    
    def _compute_aux_der_loss(self, current_aux_concat, old_aux_concat):
        """
        🎯 Auxiliary DER Loss 계산 (하위 호환성 - concat된 형태)
        
        Args:
            current_aux_concat: Tensor [B, M*C] - 현재 모델의 auxiliary logits (모달리티별 concat)
            old_aux_concat: Tensor [B, M*C] - 이전 모델의 auxiliary logits (모달리티별 concat)
            
        Note:
            - 이 메서드는 하위 호환성을 위해 유지됨
            - 새로운 코드는 _compute_aux_der_loss_modality_wise를 사용해야 함
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

 
