import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
import logging
from utils.basic_ops import ConsensusModule

class AuxiliaryHeadFusionV2_7(nn.Module):
    """
    🔥 Auxiliary Head Pretrain 전략 기반 융합 모듈 (v2_7)
    
    핵심 개선사항 (vs v2_6):
    1. ✅ Pretrain 전략: 각 task마다 처음 5 epoch 동안 auxiliary head pretrain (1:1:1 균등 가중치)
    2. ✅ Freeze 메커니즘: 각 task의 5 epoch 이후 auxiliary head 가중치 고정
    3. ✅ 균등 가중치 Fusion: 모든 epoch에서 1:1:1 균등 가중치 사용 (confidence 기반 가중치 사용 안함)
    4. ✅ 모든 task 적용: Task 0뿐만 아니라 모든 task에서 pretrain-freeze 사이클 반복
    
    학습 전략 (모든 task에서 동일):
    - **Phase 1 (Epoch 0-4)**: Auxiliary head pretrain
      * 1:1:1 균등 가중치로 fusion
      * Auxiliary head 학습 (gradient 업데이트)
      * 목적: 각 모달리티별로 좋은 classifier 학습
      
    - **Phase 2 (Epoch 5+)**: Frozen auxiliary head + uniform fusion
      * Auxiliary head 가중치 고정 (no gradient)
      * 1:1:1 균등 가중치 계속 사용 (confidence 기반 가중치 사용 안함)
      * Auxiliary head는 학습하지 않음
    
    학습 목표:
    1. 각 task마다 pretrain으로 안정적인 auxiliary head 학습
    2. 모든 epoch에서 균등한 모달리티 가중치 적용
    3. Pretrain으로 auxiliary head 안정화 후 freeze
    """
    
    def __init__(self, feature_dim, modality, dropout, num_classes=32, 
                 confidence_method="max_prob", aux_loss_weight=0.5,
                 consensus_type='avg', before_softmax=True, num_segments=8,
                 pretrain_epochs=5, energy_norm_method="zscore"):
        """
        Args:
            feature_dim: 각 모달리티 특징 차원 (1024)
            modality: 모달리티 리스트 ["RGB", "Gyro", "Acce"]
            dropout: 드롭아웃 확률
            num_classes: 클래스 수 (auxiliary head 출력 차원)
            confidence_method: 신뢰도 계산 방법 ("entropy", "max_prob", "energy", "margin", "variance", "doctor")
                ⚠️  중요: v2_7은 1:1:1 균등 가중치를 사용하므로 confidence_method는 학습에 영향을 주지 않음
                    - 학습 시: confidence_method와 무관하게 동일한 auxiliary head 학습
                    - 평가 시: confidence_method만 바꿔서 다양한 신뢰도 기반 OOD 탐지 가능
            aux_loss_weight: Auxiliary loss 가중치 (λ) - 기본값 0.5
            consensus_type: TBN consensus 방법 ('avg', 'identity')
            before_softmax: Softmax 적용 여부
            num_segments: TBN segments 수
            pretrain_epochs: Auxiliary head pretrain epoch 수 (기본값: 5)
            energy_norm_method: Energy confidence 정규화 방법 (기본값: "zscore")
                - "zscore": Z-score 정규화 (E - E_mean) / E_std → modality 간 scale 차이 제거
                - "e_uniform": E_uniform 기준 보정 sigmoid(-(E - ln(C)))
                - "raw": 정규화 없이 raw energy 사용 → sigmoid(-E)
        """
        super().__init__()
        self.modality = modality
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.confidence_method = confidence_method
        self.aux_loss_weight = aux_loss_weight
        self.energy_norm_method = energy_norm_method  # "zscore", "e_uniform", "raw"
        
        # TBN consensus 파라미터
        self.consensus_type = consensus_type
        self.before_softmax = before_softmax
        self.num_segments = num_segments
        self.reshape = True
        
        # 모달리티 인덱스 매핑
        self.modality_to_idx = {m: i for i, m in enumerate(self.modality)}
        
        # 🔥 Pretrain 메커니즘
        self.pretrain_epochs = pretrain_epochs
        self.current_epoch = 0  # 현재 epoch (외부에서 set_epoch()로 주입)
        self.current_task_id = 0
        self.auxiliary_heads_frozen = False  # Auxiliary head freeze 상태
        
        # 🎯 각 모달리티별 auxiliary head
        self.auxiliary_heads = nn.ModuleDict()
        for modality_name in self.modality:
            self.auxiliary_heads[modality_name] = nn.Linear(feature_dim, num_classes)
            normal_(self.auxiliary_heads[modality_name].weight, 0, 0.001)
            constant_(self.auxiliary_heads[modality_name].bias, 0)
        
        # 최종 융합 레이어 (Multi-modal 전용)
        if len(self.modality) <= 1:
            raise ValueError("AuxiliaryHeadFusionV2_6 requires multiple modalities")
        
        input_dim = len(self.modality) * feature_dim
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        normal_(self.fc1.weight, 0, 0.001)
        constant_(self.fc1.bias, 0)
        
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # 디버깅 플래그
        self.first_forward_per_task = {}
        self.epoch_logged = set()  # 이미 로깅된 epoch 추적 (중복 방지)
        
        # ── [Step 1.6] Modality별 energy 통계 (Z-score 정규화용) ────────
        # {modality_name: (mean: float, std: float)} 형태로 저장
        # mmeabase의 _compute_energy_stats_from_loader() 에서 채워짐
        self._energy_stats = {}  # e.g. {'RGB': (-1.5, 1.2), 'Gyro': (-5.8, 0.7)}
        
        # TBN consensus
        self.consensus = ConsensusModule(consensus_type)
        
        # Optional softmax
        if not self.before_softmax:
            self.softmax = nn.Softmax(dim=1)

    def set_energy_stats(self, energy_stats: dict):
        """
        Modality별 energy 통계를 주입 (Z-score 정규화에 사용).

        Args:
            energy_stats: {modality_name: (mean: float, std: float)}
                          e.g. {'RGB': (-1.5, 1.2), 'Gyro': (-5.8, 0.7), 'Acce': (-5.7, 0.6)}
        """
        self._energy_stats = energy_stats
        logging.info(f"[EnergyStats] Injected modality-wise energy statistics:")
        for mod, (mean, std) in energy_stats.items():
            logging.info(f"  {mod}: mean={mean:.4f}, std={std:.4f}")

    def set_epoch(self, epoch):
        """
        외부 학습 루프에서 현재 epoch 설정
        
        Args:
            epoch: 현재 epoch (0-based)
        """
        old_epoch = self.current_epoch
        self.current_epoch = epoch
        
        # 🔥 모든 task에서 pretrain 완료 시점에 auxiliary head freeze
        if epoch == self.pretrain_epochs and not self.auxiliary_heads_frozen:
            self._freeze_auxiliary_heads()
            
            if epoch not in self.epoch_logged:
                logging.info(f"")
                logging.info(f"{'='*70}")
                logging.info(f"🎉 PRETRAIN COMPLETED - AUXILIARY HEADS FROZEN!")
                logging.info(f"{'='*70}")
                logging.info(f"   Task: {self.current_task_id}")
                logging.info(f"   Epoch: {old_epoch} → {epoch}")
                logging.info(f"   Phase Transition: Pretrain → Uniform Fusion")
                logging.info(f"   Weight Mode: Uniform (1:1:1) → Uniform (1:1:1)")
                logging.info(f"   Auxiliary Heads: Training → FROZEN (no training)")
                logging.info(f"   From now on: Using uniform weights without auxiliary head training")
                logging.info(f"{'='*70}")
                self.epoch_logged.add(epoch)

    def _freeze_auxiliary_heads(self):
        """
        Auxiliary head의 파라미터를 freeze (gradient 업데이트 중단)
        """
        for modality_name, head in self.auxiliary_heads.items():
            for param in head.parameters():
                param.requires_grad = False
        
        self.auxiliary_heads_frozen = True
        logging.info(f"🔒 Auxiliary heads frozen for all modalities: {list(self.auxiliary_heads.keys())}")

    def _pick_features(self, features):
        """features 리스트에서 각 모달리티 특징 추출"""
        f_rgb = features[self.modality_to_idx['RGB']] if 'RGB' in self.modality_to_idx else None
        f_gyro = features[self.modality_to_idx['Gyro']] if 'Gyro' in self.modality_to_idx else None
        f_acce = features[self.modality_to_idx['Acce']] if 'Acce' in self.modality_to_idx else None
        return f_rgb, f_gyro, f_acce
    
    def _is_pretrain_phase(self):
        """
        현재가 pretrain 단계인지 확인
        
        Returns:
            bool: True if pretrain phase (epoch 0-4), False otherwise (epoch 5+)
        """
        # 모든 task에서 처음 5 epoch은 pretrain 적용
        return self.current_epoch < self.pretrain_epochs

    def _compute_confidence(self, logits, modality_name=None):
        """
        Auxiliary head 예측 결과로부터 신뢰도 계산
        
        Args:
            logits: [Batch, num_classes] auxiliary head 출력
            
        Returns:
            confidence: [Batch] 신뢰도 점수 (0~1)
        
        지원하는 방법:
        - "entropy": 엔트로피 기반 (전체 분포 고려, 불확실성의 역수)
        - "max_prob": 최대 확률값 (단순하지만 오분류에 취약)
        - "energy": Energy score (logits 기반, softmax 불필요)
        - "margin": Top-1과 Top-2 확률 차이 (분리도 측정)
        - "variance": 확률 분포의 분산 (낮을수록 확실)
        - "doctor": DOCtor 스타일 (max_prob - second_max_prob)
        """
        probs = F.softmax(logits, dim=1)
        
        if self.confidence_method == "entropy":
            # 엔트로피: 전체 확률 분포의 불확실성 측정
            eps = 1e-8
            entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
            max_entropy = torch.log(torch.tensor(self.num_classes, device=entropy.device, dtype=torch.float32))
            confidence = 1.0 - (entropy / max_entropy)
            
        elif self.confidence_method == "max_prob":
            # 최대 확률값 (틀린 예측에도 높은 confidence 가능)
            confidence, _ = torch.max(probs, dim=1)
        
        elif self.confidence_method == "energy":
            # Energy score: E(x) = -log(sum(exp(logits)))
            # 낮은 energy = 높은 confidence
            energy = -torch.logsumexp(logits, dim=1)

            # ── energy_norm_method 에 따른 분기 ─────────────────────────
            norm_method = self.energy_norm_method   # "zscore" | "e_uniform" | "raw"

            if self.energy_norm_method == "zscore":
                # [Step 1.6] Z-score 정규화
                # modality별 학습 분포 기반 정규화 → IMU 구조적 magnitude 차이 제거
                # confidence = -z  where z = (E - E_mean) / E_std
                stats = self._energy_stats.get(modality_name, None) if modality_name else None
                if stats is not None:
                    e_mean, e_std = stats
                    z_score = (energy - e_mean) / (e_std + 1e-8)
                    confidence = -z_score
                    norm_method = "zscore"
                else:
                    # stats 미주입 시 fallback → raw sigmoid
                    confidence = torch.sigmoid(-energy)
                    norm_method = "zscore_fallback(raw)"

            elif self.energy_norm_method == "e_uniform":
                # [Step 1.5] E_uniform 보정: sigmoid(-(E - ln(C)))
                num_classes_cur = logits.shape[1]
                energy_uniform = -torch.log(
                    torch.tensor(num_classes_cur, dtype=torch.float32, device=logits.device)
                )
                confidence = torch.sigmoid(-(energy - energy_uniform))
                norm_method = "e_uniform"

            else:
                # "raw": sigmoid(-E) — 정규화 없음
                confidence = torch.sigmoid(-energy)
                norm_method = "raw"
            # ─────────────────────────────────────────────────────────────

            # ─── [α-Diag] modality별 energy 진단 로깅 ──────────────────
            log_key = f'_energy_logged_{modality_name}'
            if not getattr(self, log_key, False):
                e_np = energy.detach().cpu().numpy()
                c_np = confidence.detach().cpu().numpy()
                num_classes = logits.shape[1]
                import numpy as np
                e_uniform_val = -np.log(num_classes)
                logging.info(
                    f"[α-Diag][{modality_name}] energy(raw):  "
                    f"mean={e_np.mean():.3f}, std={e_np.std():.3f}, "
                    f"min={e_np.min():.3f}, max={e_np.max():.3f}  "
                    f"(C={num_classes}, E_uniform={e_uniform_val:.3f})"
                )
                if norm_method == "zscore":
                    logging.info(
                        f"[α-Diag][{modality_name}] conf(zscore):  "
                        f"mean={c_np.mean():.4f}, std={c_np.std():.4f}, "
                        f"min={c_np.min():.4f}, max={c_np.max():.4f}  "
                        f"(E_mean={stats[0]:.3f}, E_std={stats[1]:.3f})"
                    )
                elif norm_method == "e_uniform":
                    logging.info(
                        f"[α-Diag][{modality_name}] conf(e_uniform):  "
                        f"mean={c_np.mean():.4f}, std={c_np.std():.4f}, "
                        f"min={c_np.min():.4f}, max={c_np.max():.4f}  "
                        f"(sigmoid(-E+lnC))"
                    )
                else:
                    logging.info(
                        f"[α-Diag][{modality_name}] conf({norm_method}):  "
                        f"mean={c_np.mean():.4f}, std={c_np.std():.4f}, "
                        f"min={c_np.min():.4f}, max={c_np.max():.4f}"
                    )
                pct_below_uniform = (e_np < e_uniform_val).mean() * 100
                pct_conf_over_half = (c_np > 0.5).mean() * 100
                logging.info(
                    f"[α-Diag][{modality_name}] 보정 진단:  "
                    f"E < E_uniform = {pct_below_uniform:.1f}%  |  "
                    f"conf > 0.5 = {pct_conf_over_half:.1f}%  "
                    f"[norm={norm_method}]"
                )
                setattr(self, log_key, True)
            # ─────────────────────────────────────────────────────────────
            
        elif self.confidence_method == "margin":
            # Margin: Top-1과 Top-2 확률의 차이
            # 두 확률이 멀수록 (margin이 클수록) 확실한 예측
            top2_probs, _ = torch.topk(probs, k=2, dim=1)
            margin = top2_probs[:, 0] - top2_probs[:, 1]
            confidence = margin  # 이미 [0, 1] 범위
            
        else:
            # Default: 균등 신뢰도
            confidence = torch.ones(logits.size(0), device=logits.device) / len(self.modality)
        
        return confidence

    def update_task(self, task_id):
        """
        새로운 task 시작 시 호출하여 task ID 업데이트
        
        Args:
            task_id: 새로운 task ID (0, 1, 2, ...)
        """
        old_task_id = self.current_task_id
        self.current_task_id = task_id
        
        if task_id not in self.first_forward_per_task:
            self.first_forward_per_task[task_id] = True
        
        # 🔥 Task 전환 시 epoch 로깅 초기화
        self.epoch_logged.clear()
        
        # 🔥 모든 task에서 pretrain 적용: epoch 리셋 + auxiliary heads unfreeze
        self.current_epoch = 0  # 각 task 시작 시 epoch 리셋
        self.auxiliary_heads_frozen = False  # Unfrozen 상태로 시작
        self._energy_logged = False  # task별 energy 진단 로깅 리셋 (legacy)
        # modality별 로깅 플래그 리셋
        for mod in getattr(self, 'modality', ['RGB', 'Gyro', 'Acce']):
            setattr(self, f'_energy_logged_{mod}', False)
        
        # Auxiliary heads unfreeze (모든 task 시작 시)
        for modality_name, head in self.auxiliary_heads.items():
            for param in head.parameters():
                param.requires_grad = True
        
        logging.info(f"")
        logging.info(f"{'='*70}")
        logging.info(f"🔥 Task {task_id}: Auxiliary Head PRETRAIN Phase")
        logging.info(f"{'='*70}")
        logging.info(f"   Previous Task: {old_task_id if task_id > 0 else 'N/A'}")
        logging.info(f"   Duration: epochs 0-{self.pretrain_epochs-1} (total {self.pretrain_epochs} epochs)")
        logging.info(f"   Weight Mode: Uniform (1:1:1)")
        logging.info(f"   Auxiliary Heads: Training (requires_grad=True) - UNFROZEN")
        logging.info(f"   Purpose: Pretrain auxiliary heads for stable confidence estimation")
        logging.info(f"   After Pretrain: Heads will be frozen at epoch {self.pretrain_epochs}")
        logging.info(f"{'='*70}")
    
    def update_auxiliary_heads(self, nb_classes):
        """
        CL에서 새로운 task 시작 시 auxiliary head의 클래스 수 업데이트
        
        Note: 이 메서드는 update_task() 이전에 호출됩니다.
              Freeze/unfreeze 상태는 update_task()에서 관리되므로,
              여기서는 가중치만 업데이트합니다.
        
        Args:
            nb_classes: 새로운 총 클래스 수
        """
        old_num_classes = self.num_classes
        self.num_classes = nb_classes
        
        for modality_name in self.modality:
            old_head = self.auxiliary_heads[modality_name]
            new_head = nn.Linear(self.feature_dim, nb_classes)
            
            # 기존 가중치 보존
            if old_num_classes > 0:
                new_head.weight.data[:old_num_classes] = old_head.weight.data
                new_head.bias.data[:old_num_classes] = old_head.bias.data
            
            # 새로운 클래스 가중치 초기화
            if nb_classes > old_num_classes:
                normal_(new_head.weight.data[old_num_classes:], 0, 0.001)
                constant_(new_head.bias.data[old_num_classes:], 0)
            
            self.auxiliary_heads[modality_name] = new_head
        
        logging.info(f"🎯 Auxiliary heads updated: {old_num_classes} → {nb_classes} classes")
        logging.info(f"   ℹ️  Freeze state will be managed by update_task()")

    def _apply_consensus_to_logits(self, aux_logits):
        """
        TBN segments 레벨 logits에 consensus 적용
        
        Args:
            aux_logits: [batch*segments, num_classes]
            
        Returns:
            consensus_logits: [batch, num_classes]
        """
        if self.num_segments <= 1:
            return aux_logits
        
        batch_size = aux_logits.size(0) // self.num_segments
        if aux_logits.size(0) % self.num_segments != 0:
            return aux_logits
        
        base_out = aux_logits
        if not self.before_softmax:
            base_out = self.softmax(base_out)
        
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        
        output = self.consensus(base_out)
        
        if self.consensus_type == 'identity':
            output = output[:, 0, :]
        else:
            output = output.squeeze(1)
        
        return output

    def forward(self, features, targets=None):
        """
        Forward pass: Pretrain + Uniform Fusion (모든 task에 적용)
        
        Phase 1 (Pretrain, epoch 0-4 in each task):
        - Auxiliary heads 학습 (gradient 업데이트)
        - 1:1:1 균등 가중치 fusion
        - Auxiliary loss 계산 및 backprop
        
        Phase 2 (Uniform Fusion, epoch 5+ in each task):
        - Auxiliary heads frozen (no gradient)
        - 1:1:1 균등 가중치 fusion (confidence 사용 안함)
        - Auxiliary loss 계산 안함
        
        Args:
            features: List[Tensor] - [f_rgb, f_gyro, f_acce]
            targets: 정답 레이블 (auxiliary loss 계산용, pretrain phase에서 필수!)
            
        Returns:
            dict: 융합된 특징 + auxiliary 정보 + loss 정보
        """
        if len(self.modality) > 1:
            # 각 모달리티 특징 분리
            f_rgb, f_gyro, f_acce = self._pick_features(features)
            modality_features = {'RGB': f_rgb, 'Gyro': f_gyro, 'Acce': f_acce}
            
            # 🎯 각 모달리티별 auxiliary head 예측 및 신뢰도 계산
            auxiliary_logits = {}
            confidences = {}
            
            # Pretrain phase에서는 gradient 계산, 이후에는 no_grad로 inference만
            is_pretrain = self._is_pretrain_phase()
            
            # Pretrain phase가 아니면 no_grad로 inference
            if not is_pretrain:
                with torch.no_grad():
                    for modality_name, feature in modality_features.items():
                        if feature is not None and modality_name in self.auxiliary_heads:
                            aux_logits_segments = self.auxiliary_heads[modality_name](feature)
                            aux_logits = self._apply_consensus_to_logits(aux_logits_segments)
                            confidence = self._compute_confidence(aux_logits, modality_name=modality_name)
                            
                            auxiliary_logits[modality_name] = aux_logits
                            confidences[modality_name] = confidence
            else:
                # Pretrain phase에서는 gradient 계산
                for modality_name, feature in modality_features.items():
                    if feature is not None and modality_name in self.auxiliary_heads:
                        aux_logits_segments = self.auxiliary_heads[modality_name](feature)
                        aux_logits = self._apply_consensus_to_logits(aux_logits_segments)
                        confidence = self._compute_confidence(aux_logits, modality_name=modality_name)
                        
                        auxiliary_logits[modality_name] = aux_logits
                        confidences[modality_name] = confidence
            
            # 🎯 v2_7: 모든 epoch에서 균등 가중치 사용 (1:1:1)
            # ⚠️  중요: confidence_tensor는 계산되지만 가중치에는 사용되지 않음
            #     - 학습 시: 1:1:1 균등 가중치로 auxiliary head 학습
            #     - 평가 시: confidence는 디버깅/분석 목적으로만 사용
            #     - OOD 탐지 시: auxiliary_logits와 confidences를 OOD detector에 전달
            if confidences:
                confidence_tensor = torch.stack(list(confidences.values()), dim=1)
                
                # 모든 phase에서 Uniform weights (1:1:1, no scaling) 사용
                weights = torch.ones_like(confidence_tensor)
                
                if is_pretrain:
                    weight_type = "Pretrain (1:1:1 uniform)"
                    phase = "Pretrain"
                else:
                    weight_type = "Uniform (1:1:1 fixed)"
                    phase = "Frozen Inference"
                
                # 🔧 디버깅: 첫 forward에서만 간단한 정보 출력
                should_debug = self.first_forward_per_task.get(self.current_task_id, False)
                
                if should_debug:
                    logging.info(f"")
                    logging.info(f"🔍 Fusion Strategy (Task {self.current_task_id}, Epoch {self.current_epoch}):")
                    logging.info(f"   🔥 Phase: {phase}")
                    logging.info(f"   🎯 Weight Type: {weight_type}")
                    logging.info(f"   🔒 Auxiliary Heads: {'Training' if is_pretrain else 'FROZEN'}")
                    logging.info(f"   ℹ️  Class-wise confidence will be logged at specific epochs")
                
                # 각 모달리티에 가중치 적용
                weighted_features = []
                weight_list = []
                
                for i, (modality_name, feature) in enumerate(modality_features.items()):
                    if feature is not None and modality_name in confidences:
                        weight = weights[:, list(confidences.keys()).index(modality_name)].unsqueeze(1)
                        
                        if self.num_segments > 1:
                            weight_expanded = weight.repeat_interleave(self.num_segments, dim=0)
                        else:
                            weight_expanded = weight
                        
                        weighted_feature = weight_expanded * feature
                        weighted_features.append(weighted_feature)
                        weight_list.append(weight.squeeze(-1))
                
                # 최종 융합
                x = torch.cat(weighted_features, dim=1)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout_layer(x)
                
            else:
                # Fallback: 균등 가중치
                x = torch.cat(list(modality_features.values()), dim=1)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout_layer(x)
                auxiliary_logits = {}
                weight_list = [torch.ones(x.size(0), device=x.device)] * len(self.modality)
            
        else:
            raise ValueError("AuxiliaryHeadFusionV2_6 requires multiple modalities")
        
        # 🎯 Auxiliary Loss 계산 (pretrain phase에서만)
        auxiliary_loss = None
        aux_loss_per_modality = {}
        
        if is_pretrain and targets is not None and auxiliary_logits:
            auxiliary_loss = 0.0
            for modality_name, aux_logits in auxiliary_logits.items():
                mod_loss = F.cross_entropy(aux_logits, targets)
                aux_loss_per_modality[modality_name] = mod_loss.item()
                auxiliary_loss += mod_loss
            auxiliary_loss /= len(auxiliary_logits)
        
        # 🔥 디버깅 정보 출력 (첫 forward에서만)
        if self.first_forward_per_task.get(self.current_task_id, False):
            logging.info(f"")
            logging.info(f"🎯 Fusion Module Summary (Task {self.current_task_id}, Epoch {self.current_epoch}):")
            logging.info(f"   🔧 Architecture: Pretrain + Uniform Fusion Strategy")
            logging.info(f"   📊 Modalities: {list(self.auxiliary_heads.keys())} ({len(self.modality)} total)")
            logging.info(f"   🎯 Weight Mode: Uniform (1:1:1) for all epochs")
            logging.info(f"   🔥 Pretrain Epochs: {self.pretrain_epochs}")
            logging.info(f"   🔒 Current Phase: {'Pretrain' if is_pretrain else 'Frozen (Uniform Fusion)'}")
            
            if targets is not None and aux_loss_per_modality:
                logging.info(f"   💰 Loss Analysis:")
                logging.info(f"      Auxiliary Loss (avg): {auxiliary_loss.item():.4f}")
                logging.info(f"      Auxiliary Weight (λ): {self.aux_loss_weight}")
                logging.info(f"      Auxiliary Loss (weighted): {(self.aux_loss_weight * auxiliary_loss).item():.4f}")
                logging.info(f"   📈 Per-Modality Aux Loss:")
                for mod_name, mod_loss in aux_loss_per_modality.items():
                    logging.info(f"      {mod_name}: {mod_loss:.4f}")
            
            logging.info(f"   📚 Task History: {list(self.first_forward_per_task.keys())}")
            logging.info(f"")
            
            self.first_forward_per_task[self.current_task_id] = False
        
        return {
            'features': x,
            'auxiliary_logits': auxiliary_logits,
            'auxiliary_loss': auxiliary_loss,
            'aux_loss_weight': self.aux_loss_weight,
            'modality_weights': torch.stack(weight_list, dim=1).detach() if weight_list else None,
            'confidences': confidences,
            'fusion_type': 'auxiliary_head_v2_7',
            'is_pretrain_phase': is_pretrain,
            'auxiliary_heads_frozen': self.auxiliary_heads_frozen
        }
    
    def compute_total_loss(self, main_loss, auxiliary_loss=None):
        """
        Main loss와 Auxiliary loss를 결합한 총 손실 계산 (모든 task에 적용)
        
        Pretrain phase (epoch 0-4 in each task):
        - total_loss = main_loss + λ * auxiliary_loss
        
        Uniform fusion phase (epoch 5+ in each task):
        - total_loss = main_loss (auxiliary_loss 사용 안함)
        
        Args:
            main_loss: 주 작업 손실 (예: CrossEntropy)
            auxiliary_loss: 보조 작업 손실 (forward에서 반환됨)
            
        Returns:
            total_loss: 결합된 총 손실
        """
        # Pretrain phase가 아니거나 auxiliary_loss가 없으면 main_loss만 사용
        if not self._is_pretrain_phase() or auxiliary_loss is None or self.aux_loss_weight == 0:
            return main_loss
        
        # Pretrain phase에서는 auxiliary_loss 추가
        total_loss = main_loss + self.aux_loss_weight * auxiliary_loss
        
        return total_loss
    
    def get_loss_breakdown(self, main_loss, auxiliary_loss=None):
        """
        손실 구성 요소별 분석 (디버깅용)
        
        Returns:
            dict: 손실 구성 요소 정보
        """
        if auxiliary_loss is None:
            auxiliary_loss = 0.0
            
        total_loss = self.compute_total_loss(main_loss, auxiliary_loss)
        
        is_pretrain = self._is_pretrain_phase()
        
        return {
            'main_loss': main_loss.item() if torch.is_tensor(main_loss) else main_loss,
            'auxiliary_loss': auxiliary_loss.item() if torch.is_tensor(auxiliary_loss) else auxiliary_loss,
            'aux_loss_weight': self.aux_loss_weight if is_pretrain else 0.0,
            'weighted_aux_loss': (self.aux_loss_weight * auxiliary_loss).item() if (is_pretrain and torch.is_tensor(auxiliary_loss)) else 0.0,
            'total_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            'aux_contribution_ratio': (self.aux_loss_weight * auxiliary_loss / total_loss).item() if (is_pretrain and torch.is_tensor(total_loss) and total_loss != 0) else 0.0,
            'is_pretrain_phase': is_pretrain
        }
    



