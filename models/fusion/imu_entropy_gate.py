import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_

class IMUEntropyGateFusion(nn.Module):
    """
    🎯 IMU Entropy 기반 지능적 가중치 융합
    
    핵심 아이디어:
    1. Gyro-Acce 각각의 정보 엔트로피 계산 (불확실성 측정)
    2. RGB는 그대로 유지, IMU 센서만 엔트로피 기반 지능적 가중치 적용
    3. FusionConcat과 동일: [RGB_1024, w_gyro*Gyro_1024, w_acce*Acce_1024] → 3072 → 512
    
    가중치 전략 (상호 보완적 엔트로피):
    - 한 센서의 엔트로피가 높고 다른 센서의 엔트로피가 낮을 때: 낮은 엔트로피(확실한) 센서를 더 활용
    - 두 센서 모두 높은 엔트로피: 두 센서를 모두 활용 (불확실하지만 상호 보완)
    - 두 센서 모두 낮은 엔트로피: 두 센서를 모두 활용 (둘 다 확실함)
    
    Entropy의 장점:
    - 정보 이론적 불확실성 측정
    - 센서 신호의 신뢰도를 정량화
    - 상호 보완적 정보량을 고려한 적응적 융합
    
    Parameters:
    - feature_dim: 각 모달리티 백본 출력 차원 (1024)
    - modality: ["RGB", "Gyro", "Acce"] 순서
    - dropout: 드롭아웃 확률
    - alpha: 게이트 민감도 (엔트로피 차이에 대한 반응 강도, 기본값 1.0)
    - lambda_gyro: Gyro 센서 기저 가중치 (기본값 1.0)
    - lambda_acce: Acce 센서 기저 가중치 (기본값 1.0)
    - shared_dim: 공통 투영 차원 (기본값 256)
    - min_weight: 최소 가중치 (기본값 0.2, 완전 차단 방지)
    - eps: 수치적 안정성을 위한 작은 값 (기본값 1e-8)
    """
    def __init__(self, feature_dim: int, modality, dropout: float = 0.5, 
                 alpha: float = 1.0, lambda_gyro: float = 1.0, lambda_acce: float = 1.0,
                 shared_dim: int = 256, min_weight: float = 0.2, eps: float = 1e-8):
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
        assert self.has_gyro and self.has_acce, "IMUEntropyGateFusion requires both Gyro and Acce modalities."
        
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

    def _compute_entropy(self, prob_dist):
        """
        확률 분포의 Shannon entropy 계산
        
        Entropy = -sum(p * log(p))
        
        Args:
            prob_dist: 확률 분포 [Batch, shared_dim]
            
        Returns:
            entropy: Shannon entropy [Batch]
        """
        # 수치적 안정성을 위해 작은 값 추가
        prob_dist = prob_dist + self.eps
        
        # Shannon entropy 계산
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + self.eps), dim=1)  # [Batch]
        
        return entropy

    def _compute_imu_weights(self, f_gyro, f_acce):
        """
        Gyro-Acce entropy 기반 지능적 가중치 계산
        
        핵심 로직 (반대 버전):
        - 공통 특징 공간으로 투영 → 확률 분포로 변환 → 각각의 entropy 계산 → 불확실성 선호 가중치 계산
        - 엔트로피 차이와 절대값을 모두 고려한 적응적 가중치
        
        Returns:
            w_gyro, w_acce: 각 센서의 가중치 [Batch, 1]
            gyro_entropy: Gyro entropy [Batch] (로깅용)
            acce_entropy: Acce entropy [Batch] (로깅용)
        """
        # 🔧 공통 특징 공간으로 투영 (1024 → shared_dim)
        gyro_proj = self.gyro_proj(f_gyro)  # [Batch, shared_dim]
        acce_proj = self.acce_proj(f_acce)  # [Batch, shared_dim]
        
        # 특징을 확률 분포로 변환 (softmax 적용)
        gyro_prob = F.softmax(gyro_proj, dim=1)  # [Batch, shared_dim]
        acce_prob = F.softmax(acce_proj, dim=1)  # [Batch, shared_dim]
        
        # 각 센서의 엔트로피 계산
        gyro_entropy = self._compute_entropy(gyro_prob)  # [Batch]
        acce_entropy = self._compute_entropy(acce_prob)  # [Batch]
        
        # 엔트로피 정규화 (0~1 범위로 스케일링)
        max_entropy = torch.log(torch.tensor(float(self.shared_dim), device=gyro_entropy.device))
        gyro_entropy_norm = gyro_entropy / max_entropy  # [Batch]
        acce_entropy_norm = acce_entropy / max_entropy  # [Batch]
        
        # 🔥 상호 보완적 가중치 전략
        # 1. 엔트로피 차이: 한 센서가 확실하고 다른 센서가 불확실할 때 차이가 큼
        entropy_diff = torch.abs(gyro_entropy_norm - acce_entropy_norm)  # [Batch]
        
        # 2. 평균 엔트로피: 두 센서의 전체적인 불확실성 수준
        avg_entropy = 0.5 * (gyro_entropy_norm + acce_entropy_norm)  # [Batch]
        
        # 🔄 반대 로직: 높은 엔트로피(불확실성)를 오히려 선호
        # 기존 가정 뒤집기: 불확실한 정보 = 더 많은 탐색 정보 = 더 사용
        
        # Gyro 가중치: Gyro가 불확실할수록 더 활용 (반대 로직)
        gyro_weight_factor = torch.where(
            entropy_diff > 0.1,  # 차이가 충분히 클 때
            gyro_entropy_norm + 0.5 * (1.0 - acce_entropy_norm),  # 불확실성 기반 (반전)
            0.5 * avg_entropy  # 전체 불확실성 기반 (반전)
        )
        
        # Acce 가중치: Acce가 불확실할수록 더 활용 (반대 로직)
        acce_weight_factor = torch.where(
            entropy_diff > 0.1,  # 차이가 충분히 클 때
            acce_entropy_norm + 0.5 * (1.0 - gyro_entropy_norm),  # 불확실성 기반 (반전)
            0.5 * avg_entropy  # 전체 불확실성 기반 (반전)
        )
        
        # sigmoid로 부드러운 가중치 변환
        w_gyro_strength = torch.sigmoid(self.alpha * gyro_weight_factor)  # [Batch]
        w_acce_strength = torch.sigmoid(self.alpha * acce_weight_factor)  # [Batch]
        
        # 최소 가중치 적용 (완전 차단 방지)
        w_gyro_strength = self.min_weight + (1.0 - self.min_weight) * w_gyro_strength
        w_acce_strength = self.min_weight + (1.0 - self.min_weight) * w_acce_strength
        
        # 최종 가중치 계산
        w_gyro = (self.lambda_gyro * w_gyro_strength).unsqueeze(1)  # [Batch, 1]
        w_acce = (self.lambda_acce * w_acce_strength).unsqueeze(1)  # [Batch, 1]
        
        return w_gyro, w_acce, gyro_entropy, acce_entropy

    def forward(self, features):
        """
        Forward pass: FusionConcat과 동일한 구조 + IMU entropy 가중치
        
        Args:
            features: List[Tensor], 각 텐서 shape [Batch, 1024]
            
        Returns:
            dict: {
                'features': Tensor[Batch, 512],        # 최종 융합된 특징
                'gyro_entropy': Tensor[Batch],         # Gyro entropy (로깅용)
                'acce_entropy': Tensor[Batch],         # Acce entropy (로깅용)
                'w_gyro': Tensor[Batch],               # Gyro 가중치 (로깅용)
                'w_acce': Tensor[Batch],               # Acce 가중치 (로깅용)
            }
        """
        # 0️⃣ 각 모달리티 특징 추출 (원본 1024차원 유지)
        f_rgb, f_gyro, f_acce = self._pick_features(features)

        # 1️⃣ IMU 센서 간 entropy 분석 및 가중치 계산
        w_gyro, w_acce, gyro_entropy, acce_entropy = self._compute_imu_weights(f_gyro, f_acce)

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
            max_entropy = torch.log(torch.tensor(float(self.shared_dim)))
            print(f"🔍 IMUEntropyGateFusion Debug:")
            print(f"   📋 Input Analysis:")
            print(f"      - Modalities: {self.modality}")
            print(f"      - Input shapes: {[f.shape for f in features]}")
            
            print(f"   📊 Entropy Analysis:")
            print(f"      - Gyro entropy: [{gyro_entropy.min().item():.4f}, {gyro_entropy.max().item():.4f}] (mean: {gyro_entropy.mean().item():.4f})")
            print(f"      - Acce entropy: [{acce_entropy.min().item():.4f}, {acce_entropy.max().item():.4f}] (mean: {acce_entropy.mean().item():.4f})")
            print(f"      - Max possible entropy: {max_entropy.item():.4f}")
            print(f"      - Gyro weights: [{w_gyro.min().item():.3f}, {w_gyro.max().item():.3f}] (mean: {w_gyro.mean().item():.3f})")
            print(f"      - Acce weights: [{w_acce.min().item():.3f}, {w_acce.max().item():.3f}] (mean: {w_acce.mean().item():.3f})")
            
            print(f"   🔧 Architecture:")
            print(f"      - Shared projection dim: {self.shared_dim}")
            print(f"      - Uncertainty metric: Shannon entropy")
            print(f"      - Weight strategy: Complementary entropy-based")
            print(f"      - Gate function: sigmoid(α * weight_factor)")
            print(f"      - Hyperparameters: α={self.alpha}, λ_gyro={self.lambda_gyro}, λ_acce={self.lambda_acce}")
            print(f"      - Weight range: [{self.min_weight:.1f}, 1.0]")
            print(f"      - Numerical stability: eps={self.eps}")
            print(f"      - Final output shape: {x.shape}")
            
            self._debug_printed = True

        return {
            "features": x,
            "gyro_entropy": gyro_entropy.detach(),    # 로깅용 (gradient 제거)
            "acce_entropy": acce_entropy.detach(),    # 로깅용
            "w_gyro": w_gyro.squeeze(1).detach(),     # 로깅용 
            "w_acce": w_acce.squeeze(1).detach(),     # 로깅용
        }
