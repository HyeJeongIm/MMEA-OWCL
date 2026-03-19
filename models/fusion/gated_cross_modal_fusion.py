import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
import logging

class GatedCrossModalFusion(nn.Module):
    """
    🎯 교차 모달리티 게이팅 융합 (Gated Cross-Modal Fusion) with Warm-up Strategy
    
    핵심 아이디어: "상호 참조 게이팅 (Cross-Reference Gating)"
    - 각 모달리티가 독립적으로 자신의 신뢰도를 평가하는 대신
    - 한 모달리티의 특징(문맥)이 다른 모달리티의 중요도를 직접 결정
    - 예: RGB 특징을 보고 "지금 IMU가 중요한 상황인가?" 판단
    
    핵심 가설:
    1. 예측 신뢰도보다 특징 벡터 패턴이 더 강건한 신호
    2. 모달리티 간 상호 보완 관계를 데이터로부터 학습 가능
    3. 문맥 기반 동적 가중치가 고정 가중치보다 효과적
    
    🔥 학습 전략: End-to-End Dynamic Fusion
    
    **처음부터 끝까지 Dynamic Fusion**:
      * Gate controllers **항상 TRAINING** (계속 학습)
      * Dynamic fusion (gate 기반 가중치, softmax로 합=1)
      * Loss: main_loss만 사용
      * 목적: Main task 성능 향상을 위해 gate가 최적의 가중치 학습
      
    **학습 메커니즘**:
    - Gate는 main loss의 gradient를 통해 학습
    - RGB가 중요한 샘플 → RGB 가중치 높게
    - IMU가 중요한 샘플 → IMU 가중치 높게
    - 샘플마다 동적으로 가중치 조정
    
    📊 Auxiliary Head와 비교:
    - Auxiliary Head: Pretrain(학습) → Frozen(inference) - 2단계
    - Gate Controller: **End-to-End Learning** - 1단계, 더 단순하고 효과적
    
    📝 Note: pretrain_epochs를 0으로 설정하여 처음부터 dynamic fusion 사용
    (Entropy regularization은 균등 분포를 학습시켜 dynamic하지 않음)
    
    구조:
    - RGB → Gate Controller → w_gyro, w_acce (IMU 가중치)
    - IMU → Gate Controller → w_rgb (RGB 가중치)  
    - 교차 가중치 적용 → Concat → 최종 융합
    
    사용법:
    ```python
    fusion = GatedCrossModalFusion(feature_dim=1024, modality=["RGB", "Gyro", "Acce"])
    
    # 새 task 시작 시 호출
    fusion.update_task(task_id=0)
    
    # 학습 루프에서 epoch 설정
    for epoch in range(num_epochs):
        fusion.set_epoch(epoch)
        # ... training ...
    
    # Forward pass
    result = fusion(features=[f_rgb, f_gyro, f_acce])
    ```
    """
    
    def __init__(self, feature_dim: int, modality: list, dropout: float = 0.5, 
                 gate_hidden_dim: int = 128, gate_activation: str = "relu",
                 pretrain_epochs: int = 0, gate_loss_weight: float = 0.5, num_classes: int = 32):
        """
        Args:
            feature_dim (int): 각 모달리티 특징 차원 (1024)
            modality (list): 모달리티 리스트 ["RGB", "Gyro", "Acce"]
            dropout (float): 드롭아웃 확률
            gate_hidden_dim (int): Gate Controller 은닉층 차원
            gate_activation (str): Gate Controller 활성화 함수 ("relu", "tanh")
            pretrain_epochs (int): Gate pretrain epoch 수 (기본값: 5)
            gate_loss_weight (float): Gate auxiliary loss 가중치 (기본값: 0.5, auxiliary head와 동일)
            num_classes (int): 클래스 수 (auxiliary head와 동일한 역할)
        """
        super().__init__()
        self.modality = modality
        self.feature_dim = feature_dim
        self.gate_hidden_dim = gate_hidden_dim
        self.num_classes = num_classes
        
        # 🔥 End-to-End Learning (pretrain 없음)
        self.pretrain_epochs = pretrain_epochs  # 0으로 설정되어 항상 dynamic fusion
        self.current_epoch = 0  # 현재 epoch (외부에서 set_epoch()로 주입)
        self.current_task_id = 0
        self.gate_controllers_frozen = False  # 항상 학습 가능
        self.aux_loss_weight = gate_loss_weight  # Auxiliary loss 가중치 (호환성 유지, 사용 안 함)
        
        # 모달리티 인덱스 매핑
        self.modality_to_idx = {m: i for i, m in enumerate(self.modality)}
        
        # 필수 모달리티 확인
        assert 'RGB' in self.modality and 'Gyro' in self.modality and 'Acce' in self.modality, \
            "GatedCrossModalFusion requires RGB, Gyro, and Acce modalities."
        
        # 활성화 함수 선택
        if gate_activation == "relu":
            self.gate_activation = nn.ReLU()
        elif gate_activation == "tanh":
            self.gate_activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {gate_activation}")
        
        # 🎯 Independent Gate Controllers (각 modality마다 별도 gate)
        # 각 gate는 다른 modality들의 정보를 보고 자신의 중요도를 판단
        # 장점: 경쟁 없이 독립적으로 학습, 공평한 학습 기회
        
        all_features_dim = feature_dim * 3  # RGB + Gyro + Acce
        
        # Gate 1: RGB 가중치 계산 (IMU 정보를 보고 RGB 중요도 판단)
        self.rgb_gate = nn.Sequential(
            nn.Linear(all_features_dim, gate_hidden_dim),
            self.gate_activation,
            nn.Dropout(p=dropout/2),
            nn.Linear(gate_hidden_dim, 1),
        )
        
        # Gate 2: Gyro 가중치 계산 (RGB + Acce 정보를 보고 Gyro 중요도 판단)
        self.gyro_gate = nn.Sequential(
            nn.Linear(all_features_dim, gate_hidden_dim),
            self.gate_activation,
            nn.Dropout(p=dropout/2),
            nn.Linear(gate_hidden_dim, 1),
        )
        
        # Gate 3: Acce 가중치 계산 (RGB + Gyro 정보를 보고 Acce 중요도 판단)
        self.acce_gate = nn.Sequential(
            nn.Linear(all_features_dim, gate_hidden_dim),
            self.gate_activation,
            nn.Dropout(p=dropout/2),
            nn.Linear(gate_hidden_dim, 1),
        )
        
        # 🎯 Gate Controller 초기화 (균등한 시작)
        for module in [self.rgb_gate, self.gyro_gate, self.acce_gate]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    normal_(layer.weight, 0, 0.01)  # 작은 가중치로 시작
                    if layer.bias is not None:
                        constant_(layer.bias, 0.0)  # 중립적 bias
        
        # 🎯 FusionConcat과 동일한 최종 FC 레이어
        input_dim = len(self.modality) * feature_dim
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        normal_(self.fc1.weight, 0, 0.001)
        constant_(self.fc1.bias, 0)
        
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # 디버깅 플래그
        self.first_forward_per_task = {}  # task별 첫 forward 여부 {task_id: bool}
        self.epoch_logged = set()  # 이미 로깅된 epoch 추적 (중복 방지)

    def set_epoch(self, epoch):
        """
        외부 학습 루프에서 현재 epoch 설정
        
        Args:
            epoch: 현재 epoch (0-based)
        """
        self.current_epoch = epoch
        # Note: pretrain_epochs=0이므로 freeze 로직 실행 안 됨

    def _freeze_gate_controllers(self):
        """
        Gate Controller의 파라미터를 freeze (gradient 업데이트 중단)
        """
        for param in self.rgb_gate.parameters():
            param.requires_grad = False
        for param in self.gyro_gate.parameters():
            param.requires_grad = False
        for param in self.acce_gate.parameters():
            param.requires_grad = False
        
        self.gate_controllers_frozen = True
        logging.info(f"🔒 Gate controllers frozen: [RGB, Gyro, Acce]")

    def _is_pretrain_phase(self):
        """
        현재가 pretrain 단계인지 확인
        
        Returns:
            bool: True if pretrain phase (epoch 0-4), False otherwise (epoch 5+)
        """
        # 모든 task에서 처음 5 epoch은 pretrain 적용
        return self.current_epoch < self.pretrain_epochs

    def update_auxiliary_heads(self, nb_classes):
        """
        CL에서 새로운 task 시작 시 클래스 수 업데이트
        
        Note: Gate controller는 auxiliary head와 달리 클래스 수에 의존하지 않지만,
              auxiliary_head_fusion과 인터페이스 호환성을 위해 제공
        
        Args:
            nb_classes: 새로운 총 클래스 수
        """
        self.num_classes = nb_classes
        logging.info(f"🎯 Gate controller updated: num_classes = {nb_classes}")

    def update_task(self, task_id):
        """
        새로운 task 시작 시 호출하여 task ID를 업데이트
        
        Args:
            task_id (int): 현재 task 번호
        """
        old_task_id = self.current_task_id
        self.current_task_id = task_id
        
        if task_id not in self.first_forward_per_task:
            self.first_forward_per_task[task_id] = True
        
        # 🔥 Task 전환 시 epoch 리셋
        self.epoch_logged.clear()
        self.current_epoch = 0
        
        # Gate controllers는 항상 학습 가능 상태 유지
        for param in self.rgb_gate.parameters():
            param.requires_grad = True
        for param in self.gyro_gate.parameters():
            param.requires_grad = True
        for param in self.acce_gate.parameters():
            param.requires_grad = True
        
        logging.info(f"")
        logging.info(f"{'='*70}")
        logging.info(f"🔥 Task {task_id}: End-to-End Dynamic Fusion")
        logging.info(f"{'='*70}")
        logging.info(f"   Previous Task: {old_task_id if task_id > 0 else 'N/A'}")
        logging.info(f"   Fusion Mode: Dynamic (gate-based, 처음부터 끝까지)")
        logging.info(f"   Gate Controllers: Training (requires_grad=True)")
        logging.info(f"   Learning: Main loss의 gradient로 gate 학습")
        logging.info(f"   Purpose: 샘플별 최적의 modality 가중치 학습")
        logging.info(f"{'='*70}")
    
    def _pick_features(self, features):
        """features 리스트에서 각 모달리티 특징 추출"""
        f_rgb = features[self.modality_to_idx['RGB']]
        f_gyro = features[self.modality_to_idx['Gyro']]
        f_acce = features[self.modality_to_idx['Acce']]
        return f_rgb, f_gyro, f_acce

    def _compute_cross_modal_weights(self, f_rgb, f_gyro, f_acce, is_pretrain):
        """
        독립적인 gate로 교차 모달리티 가중치 계산
        
        Args:
            f_rgb, f_gyro, f_acce: 각 모달리티 특징
            is_pretrain: 사용 안 함 (호환성 유지)
        
        Returns:
            dict: 각 모달리티별 가중치 및 디버깅 정보
                - w_rgb, w_gyro, w_acce: [B, 1] 각 가중치 (softmax 적용, 합=1)
        """
        # 🎯 1단계: 모든 modality 특징을 concat
        f_all = torch.cat([f_rgb, f_gyro, f_acce], dim=1)  # [B, 3072]
        
        # 🎯 2단계: 각 gate가 독립적으로 가중치 계산
        # 장점: 경쟁 없이 공평한 학습
        w_rgb_raw = self.rgb_gate(f_all)    # [B, 1]
        w_gyro_raw = self.gyro_gate(f_all)  # [B, 1]
        w_acce_raw = self.acce_gate(f_all)  # [B, 1]
        
        # 🎯 3단계: Softmax로 합이 1이 되도록 정규화
        all_logits = torch.cat([w_rgb_raw, w_gyro_raw, w_acce_raw], dim=1)  # [B, 3]
        all_weights = F.softmax(all_logits, dim=1)  # [B, 3], 각 행의 합=1
        w_rgb = all_weights[:, 0:1]   # [B, 1]
        w_gyro = all_weights[:, 1:2]  # [B, 1]
        w_acce = all_weights[:, 2:3]  # [B, 1]
        
        return {
            'w_rgb': w_rgb,
            'w_gyro': w_gyro, 
            'w_acce': w_acce,
            'rgb_gate_logits': w_rgb_raw.detach(),
            'gyro_gate_logits': w_gyro_raw.detach(),
            'acce_gate_logits': w_acce_raw.detach(),
        }

    def forward(self, features, targets=None):
        """
        Forward pass: End-to-End Dynamic Fusion
        
        **처음부터 끝까지**:
        - Gate controllers 학습 (gradient 업데이트)
        - Dynamic 가중치 fusion (gate 출력 사용, softmax로 합=1)
        - Main loss만 사용 (auxiliary loss 없음)
        
        Args:
            features: List[Tensor] - [f_rgb, f_gyro, f_acce]
            targets: 정답 레이블 (현재 사용 안 함)
            
        Returns:
            dict: 융합된 특징 + gate 정보
        """
        # 각 모달리티 특징 분리
        f_rgb, f_gyro, f_acce = self._pick_features(features)
        
        # 🎯 Gate 출력 계산 (항상 gradient 계산)
        is_pretrain = self._is_pretrain_phase()  # 항상 False (pretrain_epochs=0)
        gate_info = self._compute_cross_modal_weights(f_rgb, f_gyro, f_acce, is_pretrain)
        w_rgb_raw, w_gyro_raw, w_acce_raw = gate_info['w_rgb'], gate_info['w_gyro'], gate_info['w_acce']
        
        # 🎯 Gate 출력을 바로 사용 (softmax로 합=1 이미 보장됨)
        weights = torch.cat([w_rgb_raw, w_gyro_raw, w_acce_raw], dim=1)  # [B, 3]
        weight_type = "Dynamic (gate-based, end-to-end)"
        phase = "End-to-End Learning"
        
        # 🔧 디버깅: 첫 forward만
        should_debug = self.first_forward_per_task.get(self.current_task_id, False)
        
        if should_debug:
            logging.info(f"")
            logging.info(f"🔍 Fusion Strategy (Task {self.current_task_id}, Epoch {self.current_epoch}):")
            logging.info(f"   🔥 Phase: {phase}")
            logging.info(f"   🎯 Weight Type: {weight_type}")
            logging.info(f"   🔒 Gate Controllers: Training (end-to-end)")
            
            # 가중치 통계 (첫 샘플)
            weights_np = weights[0].detach().cpu().numpy()
            logging.info(f"   ⚖️  Weights (first sample): RGB={weights_np[0]:.3f}, "
                        f"Gyro={weights_np[1]:.3f}, Acce={weights_np[2]:.3f}")
            logging.info(f"   📊 Weight ranges: RGB[{w_rgb_raw.min():.3f}, {w_rgb_raw.max():.3f}], "
                        f"Gyro[{w_gyro_raw.min():.3f}, {w_gyro_raw.max():.3f}], "
                        f"Acce[{w_acce_raw.min():.3f}, {w_acce_raw.max():.3f}]")
        
        # 각 모달리티에 가중치 적용
        w_rgb = weights[:, 0].unsqueeze(1)   # [B, 1]
        w_gyro = weights[:, 1].unsqueeze(1)  # [B, 1]
        w_acce = weights[:, 2].unsqueeze(1)  # [B, 1]
        
        weighted_f_rgb = w_rgb * f_rgb       # [B, 1] * [B, 1024] = [B, 1024]
        weighted_f_gyro = w_gyro * f_gyro    # [B, 1] * [B, 1024] = [B, 1024]
        weighted_f_acce = w_acce * f_acce    # [B, 1] * [B, 1024] = [B, 1024]
        
        # 최종 융합
        x = torch.cat([weighted_f_rgb, weighted_f_gyro, weighted_f_acce], dim=1)  # [B, 3072]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        
        # No auxiliary loss (end-to-end learning)
        auxiliary_loss = None
        
        
        # 🔥 디버깅 정보 출력 (첫 forward에서만)
        if self.first_forward_per_task.get(self.current_task_id, False):
            logging.info(f"")
            logging.info(f"🎯 Fusion Module Summary (Task {self.current_task_id}, Epoch {self.current_epoch}):")
            logging.info(f"   🔧 Architecture: End-to-End Dynamic Fusion")
            logging.info(f"   📊 Modalities: {self.modality} ({len(self.modality)} total)")
            logging.info(f"   🧠 Gate Hidden Dim: {self.gate_hidden_dim}")
            logging.info(f"   🔒 Phase: {phase}")
            logging.info(f"   ⚙️  RGB gate: {self.feature_dim*3} → {self.gate_hidden_dim} → 1")
            logging.info(f"   ⚙️  Gyro gate: {self.feature_dim*3} → {self.gate_hidden_dim} → 1")
            logging.info(f"   ⚙️  Acce gate: {self.feature_dim*3} → {self.gate_hidden_dim} → 1")
            logging.info(f"   🔑 Independent gates: 공평한 학습 기회")
            logging.info(f"   📚 Task History: {list(self.first_forward_per_task.keys())}")
            logging.info(f"")
            
            self.first_forward_per_task[self.current_task_id] = False
        
        return {
            'features': x,
            'auxiliary_loss': auxiliary_loss,  # ← mmeabase.py가 찾는 이름!
            'aux_loss_weight': self.aux_loss_weight,  # ← mmeabase.py가 찾는 이름!
            'modality_weights': torch.stack([w_rgb.squeeze(-1), w_gyro.squeeze(-1), w_acce.squeeze(-1)], dim=1).detach(),
            'gate_info': gate_info,
            'fusion_type': 'gated_cross_modal',
            'is_pretrain_phase': is_pretrain,
            'gate_controllers_frozen': self.gate_controllers_frozen
        }
    
    def compute_total_loss(self, main_loss, auxiliary_loss=None):
        """
        Main loss와 Auxiliary loss를 결합한 총 손실 계산 (모든 task에 적용)
        
        Pretrain phase (epoch 0-4 in each task):
        - total_loss = main_loss + λ * auxiliary_loss
        
        Dynamic fusion phase (epoch 5+ in each task):
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
