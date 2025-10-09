import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
import logging

class GatedCrossModalFusion(nn.Module):
    """
    🎯 교차 모달리티 게이팅 융합 (Gated Cross-Modal Fusion)
    
    핵심 아이디어: "상호 참조 게이팅 (Cross-Reference Gating)"
    - 각 모달리티가 독립적으로 자신의 신뢰도를 평가하는 대신
    - 한 모달리티의 특징(문맥)이 다른 모달리티의 중요도를 직접 결정
    - 예: RGB 특징을 보고 "지금 IMU가 중요한 상황인가?" 판단
    
    핵심 가설:
    1. 예측 신뢰도보다 특징 벡터 패턴이 더 강건한 신호
    2. 모달리티 간 상호 보완 관계를 데이터로부터 학습 가능
    3. 문맥 기반 동적 가중치가 고정 가중치보다 효과적
    
    구조:
    - RGB → Gate Controller → w_gyro, w_acce (IMU 가중치)
    - IMU → Gate Controller → w_rgb (RGB 가중치)  
    - 교차 가중치 적용 → Concat → 최종 융합
    
    사용법:
    ```python
    fusion = GatedCrossModalFusion(feature_dim=1024, modality=["RGB", "Gyro", "Acce"])
    
    # 새 task 시작 시 호출
    fusion.update_task(task_id=0)
    
    # Forward pass
    result = fusion(features=[f_rgb, f_gyro, f_acce])
    ```
    """
    
    def __init__(self, feature_dim: int, modality: list, dropout: float = 0.5, 
                 gate_hidden_dim: int = 128, gate_activation: str = "relu"):
        """
        Args:
            feature_dim (int): 각 모달리티 특징 차원 (1024)
            modality (list): 모달리티 리스트 ["RGB", "Gyro", "Acce"]
            dropout (float): 드롭아웃 확률
            gate_hidden_dim (int): Gate Controller 은닉층 차원
            gate_activation (str): Gate Controller 활성화 함수 ("relu", "tanh")
        """
        super().__init__()
        self.modality = modality
        self.feature_dim = feature_dim
        self.gate_hidden_dim = gate_hidden_dim
        
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
        
        # 🎯 Gate Controller 1: RGB → IMU 가중치
        # "RGB 특징을 보고 IMU가 얼마나 중요한지 판단"
        # 학습을 통해 모델은 첫 번째 숫자는 'Gyro 정보가 얼마나 중요한지'를, 두 번째 숫자는 'Acce 정보가 얼마나 중요한지'를 나타내도록 훈련
        self.rgb_to_imu_gate = nn.Sequential(
            nn.Linear(feature_dim, gate_hidden_dim),
            self.gate_activation,
            nn.Dropout(p=dropout/2),  # Gate에는 절반 드롭아웃
            nn.Linear(gate_hidden_dim, 2),  # [w_gyro, w_acce]
        )
        
        # 🎯 Gate Controller 2: IMU → RGB 가중치  
        # "IMU 특징을 보고 RGB가 얼마나 중요한지 판단"
        self.imu_to_rgb_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, gate_hidden_dim),  # Gyro + Acce concat
            self.gate_activation,
            nn.Dropout(p=dropout/2),
            nn.Linear(gate_hidden_dim, 1),  # [w_rgb]
        )
        
        # 🎯 Gate Controller 초기화 (보수적 시작)
        # 초기에는 모든 모달리티를 균등하게 사용하도록 bias 설정
        for module in [self.rgb_to_imu_gate, self.imu_to_rgb_gate]:
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
        
        # 디버깅 플래그 - 각 task별 첫 forward 추적
        self.current_task = -1  # 현재 task 번호
        self.task_first_forward = {}  # task별 첫 forward 여부 {task_id: bool}

    def update_task(self, task_id):
        """
        새로운 task 시작 시 호출하여 task ID를 업데이트
        
        Args:
            task_id (int): 현재 task 번호
        """
        if task_id != self.current_task:
            self.current_task = task_id
            if task_id not in self.task_first_forward:
                self.task_first_forward[task_id] = True  # 새 task의 첫 forward 예약
    
    def _pick_features(self, features):
        """features 리스트에서 각 모달리티 특징 추출"""
        f_rgb = features[self.modality_to_idx['RGB']]
        f_gyro = features[self.modality_to_idx['Gyro']]
        f_acce = features[self.modality_to_idx['Acce']]
        return f_rgb, f_gyro, f_acce

    def _compute_cross_modal_weights(self, f_rgb, f_gyro, f_acce):
        """
        교차 모달리티 가중치 계산
        
        Returns:
            dict: 각 모달리티별 가중치 및 디버깅 정보
        """
        batch_size = f_rgb.size(0)
        
        # 🎯 1단계: RGB → IMU 가중치 계산
        # "RGB 특징을 보니, 지금 IMU가 얼마나 중요한가?"
        imu_gate_logits = self.rgb_to_imu_gate(f_rgb)  # [B, 2]
        w_gyro_raw, w_acce_raw = imu_gate_logits.chunk(2, dim=1)  # 각각 [B, 1]
        
        # 🎯 2단계: IMU → RGB 가중치 계산  
        # "IMU 특징을 보니, 지금 RGB가 얼마나 중요한가?"
        f_imu_concat = torch.cat([f_gyro, f_acce], dim=1)  # [B, 2048]
        rgb_gate_logits = self.imu_to_rgb_gate(f_imu_concat)  # [B, 1]
        w_rgb_raw = rgb_gate_logits
        
        # 🎯 3단계: Sigmoid로 0~1 범위 가중치 생성
        w_rgb = torch.sigmoid(w_rgb_raw)    # [B, 1]
        w_gyro = torch.sigmoid(w_gyro_raw)  # [B, 1] 
        w_acce = torch.sigmoid(w_acce_raw)  # [B, 1]
        
        return {
            'w_rgb': w_rgb,
            'w_gyro': w_gyro, 
            'w_acce': w_acce,
            'rgb_gate_logits': rgb_gate_logits.detach(),
            'imu_gate_logits': imu_gate_logits.detach(),
        }

    def forward(self, features, targets=None):
        """
        Args:
            features (list): 모달리티 순서에 따른 특징 벡터 리스트
            targets (torch.Tensor, optional): 정답 레이블 (현재 사용 안 함)
            
        Returns:
            dict: 융합된 특징 및 게이팅 정보
        """
        # 1️⃣ 특징 벡터 분리
        f_rgb, f_gyro, f_acce = self._pick_features(features)
        
        # 2️⃣ 교차 모달리티 가중치 계산
        gate_info = self._compute_cross_modal_weights(f_rgb, f_gyro, f_acce)
        w_rgb, w_gyro, w_acce = gate_info['w_rgb'], gate_info['w_gyro'], gate_info['w_acce']
        
        # 3️⃣ 가중치 적용 (Broadcasting)
        weighted_f_rgb = w_rgb * f_rgb      # [B, 1] * [B, 1024] = [B, 1024]
        weighted_f_gyro = w_gyro * f_gyro   # [B, 1] * [B, 1024] = [B, 1024]  
        weighted_f_acce = w_acce * f_acce   # [B, 1] * [B, 1024] = [B, 1024]
        
        # 4️⃣ FusionConcat과 동일한 방식으로 결합
        x = torch.cat([weighted_f_rgb, weighted_f_gyro, weighted_f_acce], dim=1)  # [B, 3072]
        x = self.fc1(x)      # [B, 512]
        x = self.relu(x)
        x = self.dropout_layer(x)
        
        # 5️⃣ 디버깅 정보 출력 (각 task의 첫 번째 forward에서만)
        if self.current_task in self.task_first_forward and self.task_first_forward[self.current_task]:
            logging.info(f"🎯 GatedCrossModalFusion Debug (Task {self.current_task}):")
            logging.info(f"   Modality count: {len(self.modality)}")
            logging.info(f"   Gate hidden dim: {self.gate_hidden_dim}")
            logging.info(f"   RGB→IMU gate: {self.feature_dim} → {self.gate_hidden_dim} → 2")
            logging.info(f"   IMU→RGB gate: {self.feature_dim*2} → {self.gate_hidden_dim} → 1")
            logging.info(f"   Weight ranges: RGB[{w_rgb.min():.3f}, {w_rgb.max():.3f}], "
                        f"Gyro[{w_gyro.min():.3f}, {w_gyro.max():.3f}], "
                        f"Acce[{w_acce.min():.3f}, {w_acce.max():.3f}]")
            logging.info(f"   Architecture: Cross-Modal Gating → Weighted Concat → FC")
            self.task_first_forward[self.current_task] = False  # 해당 task의 첫 forward 완료
        
        # 6️⃣ 결과 반환
        return {
            'features': x,
            'modality_weights': torch.cat([w_rgb, w_gyro, w_acce], dim=1).detach(),  # [B, 3]
            'gate_info': gate_info,  # 상세 게이팅 정보
        }
