import torch
import torch.nn as nn
import torch.nn.functional as F

class IMUCosineGateFusion(nn.Module):
    """
    🎯 Gyro/Acce 간 코사인 유사도 기반 IMU 게이팅 → RGB와 선택적 결합하여 512-D 특징 반환
    
    핵심 아이디어:
    1. Gyro와 Acce의 코사인 유사도를 계산
    2. 유사도가 낮을수록 (서로 다를수록) 두 센서를 모두 강하게 활용
    3. RGB와 IMU를 선택적으로 결합
    
    Parameters:
    - modality: ["RGB", "Gyro", "Acce"] 순서를 그대로 사용
    - feature_dim: 각 모달리티 백본 출력 차원 (예: 1024)
    - dropout: 최종 드롭아웃 확률
    - alpha: (1 - cos) 스케일 (게이트 온도) - 클수록 차이에 민감
    - gamma: RGB와 결합 시 IMU 전체 세기 - 작을수록 RGB 위주
    - lambda_gyro/lambda_acce: IMU 내부에서 Gyro/Acce의 기저 비율
    """
    def __init__(self, feature_dim: int, modality, dropout: float = 0.5,
                 alpha: float = 5.0, gamma: float = 0.3,
                 lambda_gyro: float = 1.0, lambda_acce: float = 1.1):
        super().__init__()
        self.modality = modality
        self.feature_dim = feature_dim
        self.out_dim = 512

        # 모달리티 존재 여부 확인
        self.has_rgb  = ('RGB'  in modality)
        self.has_gyro = ('Gyro' in modality)
        self.has_acce = ('Acce' in modality)
        
        # IMU 게이팅은 Gyro와 Acce가 모두 필요
        assert self.has_gyro and self.has_acce, "IMUCosineGateFusion requires both Gyro and Acce modalities."

        # 1️⃣ 모달리티별 1024→512 투영 레이어
        self.proj_gyro = nn.Sequential(
            nn.Linear(feature_dim, self.out_dim), 
            nn.ReLU(inplace=True)
        )
        self.proj_acce = nn.Sequential(
            nn.Linear(feature_dim, self.out_dim), 
            nn.ReLU(inplace=True)
        )
        if self.has_rgb:
            self.proj_rgb = nn.Sequential(
                nn.Linear(feature_dim, self.out_dim), 
                nn.ReLU(inplace=True)
            )

        # 2️⃣ RGB와 IMU 최종 결합층
        if self.has_rgb:
            # RGB + IMU 결합: [512 + 512] → 512
            self.fuse_out = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.out_dim * 2, self.out_dim),
                nn.ReLU(inplace=True)
            )
        else:
            # IMU만 사용: 512 → 512 (Identity)
            self.fuse_out = nn.Sequential(
                nn.Dropout(dropout),
                nn.Identity()
            )

        # 3️⃣ 하이퍼파라미터 저장
        self.alpha = alpha              # 게이트 온도 (차이에 대한 민감도)
        self.gamma = gamma              # IMU 전체 강도
        self.lambda_gyro = lambda_gyro  # Gyro 기저 가중치
        self.lambda_acce = lambda_acce  # Acce 기저 가중치
        
        # 디버깅 플래그
        self._debug_printed = False

    def _pick_features(self, features):
        """
        features 리스트에서 각 모달리티 텐서를 정확한 순서로 추출
        
        TBN backbone은 self.modality 순서대로 특징을 반환합니다:
        - self.modality = ["RGB", "Gyro", "Acce"] → features = [rgb_feat, gyro_feat, acce_feat]
        - self.modality = ["Gyro", "Acce"] → features = [gyro_feat, acce_feat]
        """
        # modality 이름을 인덱스로 매핑
        modality_to_idx = {modality: i for i, modality in enumerate(self.modality)}
        
        # 각 모달리티가 존재하는 경우에만 해당 인덱스에서 추출
        f_rgb  = features[modality_to_idx['RGB']]  if self.has_rgb  else None
        f_gyro = features[modality_to_idx['Gyro']] if self.has_gyro else None
        f_acce = features[modality_to_idx['Acce']] if self.has_acce else None
        
        return f_rgb, f_gyro, f_acce

    def forward(self, features):
        """
        Forward pass: 코사인 유사도 기반 IMU 게이팅 + RGB 결합
        
        Args:
            features: List[Tensor], 각 텐서 shape [Batch, feature_dim]
            
        Returns:
            dict: {
                'features': Tensor[Batch, 512],     # 최종 융합된 특징
                'cosine': Tensor[Batch],            # Gyro-Acce 코사인 유사도 (로깅용)
                'w_gyro': Tensor[Batch],            # Gyro 가중치 (로깅용)
                'w_acce': Tensor[Batch],            # Acce 가중치 (로깅용)
            }
        """
        # 0️⃣ 각 모달리티 특징 추출
        f_rgb, f_gyro, f_acce = self._pick_features(features)

        # 1️⃣ 각 모달리티를 512차원으로 투영
        g = self.proj_gyro(f_gyro)   # [Batch, 512]
        a = self.proj_acce(f_acce)   # [Batch, 512]
        r = self.proj_rgb(f_rgb) if self.has_rgb else None  # [Batch, 512] or None

        # 2️⃣ Gyro-Acce 코사인 유사도 계산
        g_normalized = F.normalize(g, dim=1)  # L2 정규화
        a_normalized = F.normalize(a, dim=1)  # L2 정규화
        cosine_sim = torch.clamp((g_normalized * a_normalized).sum(dim=1), -1.0, 1.0)  # [Batch]
        
        # 차이 계산: 유사도가 낮을수록 (서로 다를수록) 큰 값
        difference = 1.0 - cosine_sim  # 범위: [0, 2]

        # 3️⃣ IMU 내부 게이팅 가중치 계산
        # 차이가 클수록 두 센서를 모두 강하게 활용
        gate_strength = torch.sigmoid(self.alpha * difference)  # [Batch]
        
        w_gyro = (self.lambda_gyro * gate_strength).unsqueeze(1)  # [Batch, 1]
        w_acce = (self.lambda_acce * gate_strength).unsqueeze(1)  # [Batch, 1]

        # 4️⃣ 가중 IMU 특징 생성
        f_imu = w_gyro * g + w_acce * a  # [Batch, 512]

        # 5️⃣ RGB와 IMU 최종 결합
        if self.has_rgb and r is not None:
            # RGB + IMU 결합
            combined = torch.cat([r, self.gamma * f_imu], dim=1)  # [Batch, 1024]
            final_features = self.fuse_out(combined)  # [Batch, 512]
        else:
            # IMU만 사용
            final_features = self.fuse_out(self.gamma * f_imu)  # [Batch, 512]

        # 6️⃣ 디버그 정보 출력 (첫 번째 forward에서만)
        if not self._debug_printed:
            print(f"🔍 IMUCosineGateFusion Debug:")
            print(f"   Modalities: {self.modality}")
            print(f"   Feature mapping: {[(i, mod) for i, mod in enumerate(self.modality)]}")
            print(f"   Input shapes: {[f.shape for f in features]}")
            print(f"   RGB shape: {f_rgb.shape if f_rgb is not None else 'None'}")
            print(f"   Gyro shape: {f_gyro.shape if f_gyro is not None else 'None'}")
            print(f"   Acce shape: {f_acce.shape if f_acce is not None else 'None'}")
            print(f"   Cosine similarity range: [{cosine_sim.min().item():.3f}, {cosine_sim.max().item():.3f}]")
            print(f"   Gate strength range: [{gate_strength.min().item():.3f}, {gate_strength.max().item():.3f}]")
            print(f"   Final output shape: {final_features.shape}")
            print(f"   Hyperparameters: alpha={self.alpha}, gamma={self.gamma}, λ_gyro={self.lambda_gyro}, λ_acce={self.lambda_acce}")
            self._debug_printed = True

        return {
            "features": final_features,
            "cosine": cosine_sim.detach(),            # 로깅용 (gradient 제거)
            "w_gyro": w_gyro.squeeze(1).detach(),     # 로깅용 
            "w_acce": w_acce.squeeze(1).detach(),     # 로깅용
        }
