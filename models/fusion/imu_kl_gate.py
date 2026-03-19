import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_

class IMUKLGateFusion(nn.Module):
    """
    🎯 IMU KL Divergence 기반 지능적 가중치 융합
    
    핵심 아이디어:
    1. Gyro-Acce 간 KL divergence 계산 (확률 분포의 차이)
    2. RGB는 그대로 유지, IMU 센서만 지능적 가중치 적용
    3. FusionConcat과 동일: [RGB_1024, w_gyro*Gyro_1024, w_acce*Acce_1024] → 3072 → 512
    
    가중치 전략:
    - KL divergence가 클수록 (분포가 다를수록): 두 센서를 강력하게 활용 (≈1.0)
    - KL divergence가 작을수록 (분포가 비슷할수록): 두 센서를 약하게 활용 (≈0.1)
    
    KL divergence의 장점:
    - 확률 분포 간의 정보 이론적 차이를 측정
    - 비대칭적 특성으로 방향성 있는 차이 포착
    - 센서 신호의 불확실성과 정보량 차이를 고려
    
    Parameters:
    - feature_dim: 각 모달리티 백본 출력 차원 (1024)
    - modality: ["RGB", "Gyro", "Acce"] 순서
    - dropout: 드롭아웃 확률
    - alpha: 게이트 민감도 (KL divergence에 대한 반응 강도, 기본값 0.5)
    - lambda_gyro: Gyro 센서 기저 가중치 (기본값 1.0)
    - lambda_acce: Acce 센서 기저 가중치 (기본값 1.0)
    - shared_dim: 공통 투영 차원 (기본값 256)
    - min_weight: 최소 가중치 (기본값 0.1, 완전 차단 방지)
    - eps: 수치적 안정성을 위한 작은 값 (기본값 1e-8)
    """
    def __init__(self, feature_dim: int, modality, dropout: float = 0.5, 
                 alpha: float = 0.5, lambda_gyro: float = 1.0, lambda_acce: float = 1.0,
                 shared_dim: int = 256, min_weight: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.modality = modality
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.lambda_gyro = lambda_gyro
        self.lambda_acce = lambda_acce
        self.shared_dim = shared_dim
        self.min_weight = min_weight
        self.eps = eps
        
        # 모달리티 존재 여부 확인
        self.has_rgb  = ('RGB'  in modality)
        self.has_gyro = ('Gyro' in modality)
        self.has_acce = ('Acce' in modality)
        
        # IMU 분석은 Gyro와 Acce가 모두 필요
        assert self.has_gyro and self.has_acce, "IMUKLGateFusion requires both Gyro and Acce modalities."
        
        # 🔧 공통 특징 공간으로 투영하는 레이어들
        self.gyro_proj = nn.Linear(feature_dim, shared_dim)
        self.acce_proj = nn.Linear(feature_dim, shared_dim)
        
        # 투영 레이어 초기화
        normal_(self.gyro_proj.weight, 0, 0.001)
        constant_(self.gyro_proj.bias, 0)
        normal_(self.acce_proj.weight, 0, 0.001)
        constant_(self.acce_proj.bias, 0)

        # 🎯 FusionConcat과 동일한 구조: 3072 → 512 (공정한 비교)
        if len(self.modality) > 1:
            input_dim = len(self.modality) * feature_dim  # 3 * 1024 = 3072
            self.fc1 = nn.Linear(input_dim, 512)
            self.relu = nn.ReLU()
            
            # FusionConcat과 동일한 weight 초기화
            normal_(self.fc1.weight, 0, 0.001)
            constant_(self.fc1.bias, 0)
        
        # 드롭아웃 레이어 (FusionConcat과 동일)
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # 디버깅 플래그
        self._debug_printed = False

    def _pick_features(self, features):
        """
        features 리스트에서 각 모달리티 텐서를 정확한 순서로 추출
        """
        # modality 이름을 인덱스로 매핑
        modality_to_idx = {modality: i for i, modality in enumerate(self.modality)}
        
        # 각 모달리티가 존재하는 경우에만 해당 인덱스에서 추출
        f_rgb  = features[modality_to_idx['RGB']]  if self.has_rgb  else None
        f_gyro = features[modality_to_idx['Gyro']] if self.has_gyro else None
        f_acce = features[modality_to_idx['Acce']] if self.has_acce else None
        
        return f_rgb, f_gyro, f_acce

    def _compute_imu_weights(self, f_gyro, f_acce):
        """
        Gyro-Acce KL divergence 기반 지능적 가중치 계산
        
        핵심 로직 (반대 버전):
        - 공통 특징 공간으로 투영 → 확률 분포로 변환 → KL divergence 계산 → 반전 → sigmoid로 게이트 강도 계산
        - KL divergence가 작을수록 (분포가 유사할수록) 두 센서를 모두 강하게 활용
        
        Returns:
            w_gyro, w_acce: 각 센서의 가중치 [Batch, 1]
            kl_div: KL divergence [Batch] (로깅용)
        """
        # 🔧 공통 특징 공간으로 투영 (1024 → shared_dim)
        gyro_proj = self.gyro_proj(f_gyro)  # [Batch, shared_dim]
        acce_proj = self.acce_proj(f_acce)  # [Batch, shared_dim]
        
        # 특징을 확률 분포로 변환 (softmax 적용)
        gyro_prob = F.softmax(gyro_proj, dim=1)  # [Batch, shared_dim]
        acce_prob = F.softmax(acce_proj, dim=1)  # [Batch, shared_dim]
        
        # 수치적 안정성을 위해 작은 값 추가
        gyro_prob = gyro_prob + self.eps
        acce_prob = acce_prob + self.eps
        
        # 재정규화
        gyro_prob = gyro_prob / gyro_prob.sum(dim=1, keepdim=True)
        acce_prob = acce_prob / acce_prob.sum(dim=1, keepdim=True)
        
        # KL divergence 계산: KL(P||Q) = sum(P * log(P/Q))
        # 양방향 KL divergence의 평균을 사용 (대칭적 측정)
        kl_gyro_acce = torch.sum(gyro_prob * torch.log(gyro_prob / acce_prob + self.eps), dim=1)  # [Batch]
        kl_acce_gyro = torch.sum(acce_prob * torch.log(acce_prob / gyro_prob + self.eps), dim=1)  # [Batch]
        
        # 대칭적 KL divergence (Jensen-Shannon divergence의 근사)
        kl_div = 0.5 * (kl_gyro_acce + kl_acce_gyro)  # [Batch]
        
        # 🔄 반대 로직: KL divergence가 작을수록 (더 유사할수록) 강한 가중치
        # 기존 가정 뒤집기: 유사한 분포 = 신뢰도 높음 = 더 사용
        # KL divergence를 반전시켜서 유사할수록 큰 값이 되도록 함
        max_kl = 10.0  # 적당한 최대값으로 정규화
        inverted_kl = torch.clamp(max_kl - kl_div, min=0.0) / max_kl  # [0, 1] 범위로 정규화
        gate_strength = torch.sigmoid(self.alpha * inverted_kl)  # [Batch]
        
        # 최소 가중치 적용 (완전 차단 방지)
        gate_strength = self.min_weight + (1.0 - self.min_weight) * gate_strength
        
        # 가중치 계산 (기본값 1.0으로 동일, 필요시 조정 가능)
        w_gyro = (self.lambda_gyro * gate_strength).unsqueeze(1)  # [Batch, 1]
        w_acce = (self.lambda_acce * gate_strength).unsqueeze(1)  # [Batch, 1]
        
        return w_gyro, w_acce, kl_div

    def forward(self, features):
        """
        Forward pass: FusionConcat과 동일한 구조 + IMU KL divergence 가중치
        
        Args:
            features: List[Tensor], 각 텐서 shape [Batch, 1024]
            
        Returns:
            dict: {
                'features': Tensor[Batch, 512],        # 최종 융합된 특징
                'kl_div': Tensor[Batch],               # Gyro-Acce KL divergence (로깅용)
                'w_gyro': Tensor[Batch],               # Gyro 가중치 (로깅용)
                'w_acce': Tensor[Batch],               # Acce 가중치 (로깅용)
            }
        """
        # 0️⃣ 각 모달리티 특징 추출 (원본 1024차원 유지)
        f_rgb, f_gyro, f_acce = self._pick_features(features)

        # 1️⃣ IMU 센서 간 KL divergence 분석 및 가중치 계산
        w_gyro, w_acce, kl_div = self._compute_imu_weights(f_gyro, f_acce)

        # 2️⃣ 가중치 적용: RGB는 그대로, IMU만 지능적 가중치
        weighted_gyro = w_gyro * f_gyro  # [Batch, 1024]
        weighted_acce = w_acce * f_acce  # [Batch, 1024]

        # 3️⃣ FusionConcat과 동일한 방식으로 결합
        if len(self.modality) > 1:
            # RGB + 가중치 적용된 IMU 센서들 결합
            x = torch.cat([f_rgb, weighted_gyro, weighted_acce], dim=1)  # [Batch, 3072]
            x = self.fc1(x)      # [Batch, 512]
            x = self.relu(x)
            x = self.dropout_layer(x)
        else:
            # Single modality (이론적으로는 발생하지 않음)
            x = features[0]
            x = self.dropout_layer(x)

        # 4️⃣ 디버그 정보 출력 (첫 번째 forward에서만)
        if not self._debug_printed:
            print(f"🔍 IMUKLGateFusion Debug:")
            print(f"   📋 Input Analysis:")
            print(f"      - Modalities: {self.modality}")
            print(f"      - Input shapes: {[f.shape for f in features]}")
            
            print(f"   📊 KL Divergence Analysis:")
            print(f"      - KL divergence: [{kl_div.min().item():.4f}, {kl_div.max().item():.4f}] (mean: {kl_div.mean().item():.4f})")
            print(f"      - Gate strength: [{w_gyro.min().item():.3f}, {w_gyro.max().item():.3f}] (mean: {w_gyro.mean().item():.3f})")
            
            print(f"   🔧 Architecture:")
            print(f"      - Shared projection dim: {self.shared_dim}")
            print(f"      - Distance metric: Symmetric KL divergence")
            print(f"      - Probability conversion: softmax(projection)")
            print(f"      - Gate function: {self.min_weight:.1f} + {1.0 - self.min_weight:.1f} * sigmoid(α * KL_div)")
            print(f"      - Hyperparameters: α={self.alpha}, λ_gyro={self.lambda_gyro}, λ_acce={self.lambda_acce}")
            print(f"      - Weight range: [{self.min_weight:.1f}, 1.0]")
            print(f"      - Numerical stability: eps={self.eps}")
            print(f"      - Final output shape: {x.shape}")
            
            self._debug_printed = True

        return {
            "features": x,
            "kl_div": kl_div.detach(),                # 로깅용 (gradient 제거)
            "w_gyro": w_gyro.squeeze(1).detach(),     # 로깅용 
            "w_acce": w_acce.squeeze(1).detach(),     # 로깅용
        }
