import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_

class IMUCosineGateFusion(nn.Module):
    """
    🎯 IMU 시너지 기반 지능적 가중치 융합 (FusionConcat과 동일한 구조로 공정한 비교)
    
    핵심 아이디어:
    1. Gyro-Acce 코사인 유사도 계산 (시너지 분석)
    2. RGB는 그대로 유지, IMU 센서만 지능적 가중치 적용
    3. FusionConcat과 동일: [RGB_1024, w_gyro*Gyro_1024, w_acce*Acce_1024] → 3072 → 512
    
    가중치 전략 (반대 로직 - 신뢰도 기반):
    - 코사인 유사도가 높을수록 (높은 신뢰도): 두 센서를 강력하게 활용 (≈1.0)
    - 코사인 유사도가 낮을수록 (낮은 신뢰도): 두 센서를 거의 차단 (≈0.001)
    
    Parameters:
    - feature_dim: 각 모달리티 백본 출력 차원 (1024)
    - modality: ["RGB", "Gyro", "Acce"] 순서
    - dropout: 드롭아웃 확률
    - alpha: 게이트 민감도 (차이에 대한 반응 강도, 기본값 15.0)
    - lambda_gyro: Gyro 센서 기저 가중치 (기본값 1.0)
    - lambda_acce: Acce 센서 기저 가중치 (기본값 1.0)
    - shared_dim: 공통 투영 차원 (기본값 256)
    - bias_offset: 게이트 bias (기본값 -7.0, 비슷할 때 극단적 차단용)
    """
    def __init__(self, feature_dim: int, modality, dropout: float = 0.5, 
                 alpha: float = 15.0, lambda_gyro: float = 1.0, lambda_acce: float = 1.0,
                 shared_dim: int = 256, bias_offset: float = -7.0):
        super().__init__()
        self.modality = modality
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.lambda_gyro = lambda_gyro
        self.lambda_acce = lambda_acce
        self.shared_dim = shared_dim
        self.bias_offset = bias_offset
        
        # 모달리티 존재 여부 확인
        self.has_rgb  = ('RGB'  in modality)
        self.has_gyro = ('Gyro' in modality)
        self.has_acce = ('Acce' in modality)
        
        # IMU 시너지 분석은 Gyro와 Acce가 모두 필요
        assert self.has_gyro and self.has_acce, "IMUSynergyFusion requires both Gyro and Acce modalities."
        
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
        
        TBN backbone은 self.modality 순서대로 특징을 반환합니다:
        - self.modality = ["RGB", "Gyro", "Acce"] → features = [rgb_feat, gyro_feat, acce_feat]
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
        Gyro-Acce 코사인 유사도 기반 지능적 가중치 계산
        
        핵심 로직 (개선된 버전):
        - 공통 특징 공간으로 투영 → 코사인 유사도 계산 → 차이 계산 → sigmoid로 게이트 강도 계산
        - 차이가 클수록 (서로 다를수록) 두 센서를 모두 강하게 활용
        
        Returns:
            w_gyro, w_acce: 각 센서의 가중치 [Batch, 1]
            cosine_sim: 코사인 유사도 [Batch] (로깅용)
        """
        # 🔧 공통 특징 공간으로 투영 (1024 → shared_dim)
        gyro_proj = self.gyro_proj(f_gyro)  # [Batch, shared_dim]
        acce_proj = self.acce_proj(f_acce)  # [Batch, shared_dim]
        
        # L2 정규화 (투영된 공간에서)
        gyro_norm = F.normalize(gyro_proj, dim=1)  # [Batch, shared_dim]
        acce_norm = F.normalize(acce_proj, dim=1)  # [Batch, shared_dim]
        
        # 코사인 유사도 계산 (이제 같은 의미 공간에서!)
        cosine_sim = torch.clamp((gyro_norm * acce_norm).sum(dim=1), -1.0, 1.0)  # [Batch]
        
        # 🔄 반대 로직: 유사도가 높을수록 (서로 비슷할수록) 더 활용
        # 기존 가정 뒤집기: 비슷한 정보 = 신뢰도 높음 = 더 사용
        similarity = cosine_sim  # 범위: [-1, 1] → [0, 2]로 변환
        similarity_normalized = (similarity + 1.0)  # 범위: [0, 2]
        
        # 🔥 반대 게이트 강도 계산: 유사도가 높을수록 강하게 활용!
        # 비슷할 때 (similarity≈2): sigmoid(15.0*2 + (-7.0)) = sigmoid(23.0) ≈ 1.000
        # 다를 때 (similarity≈0): sigmoid(15.0*0 + (-7.0)) = sigmoid(-7.0) ≈ 0.001
        gate_input = self.alpha * similarity_normalized + self.bias_offset
        gate_strength = torch.sigmoid(gate_input)  # [Batch]
        
        # 가중치 계산 (기본값 1.0으로 동일, 필요시 조정 가능)
        w_gyro = (self.lambda_gyro * gate_strength).unsqueeze(1)  # [Batch, 1]
        w_acce = (self.lambda_acce * gate_strength).unsqueeze(1)  # [Batch, 1]
        
        return w_gyro, w_acce, cosine_sim

    def forward(self, features):
        """
        Forward pass: FusionConcat과 동일한 구조 + IMU 지능적 가중치
        
        Args:
            features: List[Tensor], 각 텐서 shape [Batch, 1024]
            
        Returns:
            dict: {
                'features': Tensor[Batch, 512],     # 최종 융합된 특징
                'cosine': Tensor[Batch],            # Gyro-Acce 코사인 유사도 (로깅용)
                'w_gyro': Tensor[Batch],            # Gyro 가중치 (로깅용)
                'w_acce': Tensor[Batch],            # Acce 가중치 (로깅용)
            }
        """
        # 0️⃣ 각 모달리티 특징 추출 (원본 1024차원 유지)
        f_rgb, f_gyro, f_acce = self._pick_features(features)

        # 1️⃣ IMU 센서 간 시너지 분석 및 가중치 계산
        w_gyro, w_acce, cosine_sim = self._compute_imu_weights(f_gyro, f_acce)

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
            print(f"🔍 IMUCosineGateFusion Debug:")
            print(f"   📋 Input Analysis:")
            print(f"      - Modalities: {self.modality}")
            print(f"      - Input shapes: {[f.shape for f in features]}")
            print(f"      - Feature mapping: {[(i, mod) for i, mod in enumerate(self.modality)]}")
            
            print(f"   🎯 Extracted Features:")
            print(f"      - RGB shape: {f_rgb.shape if f_rgb is not None else 'None'}")
            print(f"      - Gyro shape: {f_gyro.shape if f_gyro is not None else 'None'}")
            print(f"      - Acce shape: {f_acce.shape if f_acce is not None else 'None'}")
            
            print(f"   📊 Synergy Analysis:")
            print(f"      - Cosine similarity: [{cosine_sim.min().item():.3f}, {cosine_sim.max().item():.3f}] (mean: {cosine_sim.mean().item():.3f})")
            print(f"      - Gate strength: [{w_gyro.min().item():.3f}, {w_gyro.max().item():.3f}] (mean: {w_gyro.mean().item():.3f})")
            
            print(f"   🔧 Architecture:")
            print(f"      - Shared projection dim: {self.shared_dim}")
            print(f"      - Gyro projection: 1024 → {self.shared_dim}")
            print(f"      - Acce projection: 1024 → {self.shared_dim}")
            print(f"      - Weighted concat shape: {x.shape if len(self.modality) > 1 else 'Single modality'}")
            print(f"      - Final output shape: {x.shape}")
            print(f"      - Hyperparameters: α={self.alpha}, λ_gyro={self.lambda_gyro}, λ_acce={self.lambda_acce}")
            print(f"      - Extreme gating: bias_offset={self.bias_offset}")
            print(f"      - Gate function: sigmoid(α*difference + bias)")
            print(f"      - Similar (diff≈0): sigmoid({self.alpha}*0 + {self.bias_offset}) ≈ {torch.sigmoid(torch.tensor(self.bias_offset)).item():.4f}")
            print(f"      - Different (diff≈2): sigmoid({self.alpha}*2 + {self.bias_offset}) ≈ {torch.sigmoid(torch.tensor(self.alpha*2 + self.bias_offset)).item():.4f}")
            print(f"      - Structure: RGB_1024 + w_gyro*Gyro_1024 + w_acce*Acce_1024 → 3072 → 512")
            print(f"      - Cosine similarity computed in shared {self.shared_dim}D space")
            
            self._debug_printed = True

        return {
            "features": x,
            "cosine": cosine_sim.detach(),            # 로깅용 (gradient 제거)
            "w_gyro": w_gyro.squeeze(1).detach(),     # 로깅용 
            "w_acce": w_acce.squeeze(1).detach(),     # 로깅용
        }
