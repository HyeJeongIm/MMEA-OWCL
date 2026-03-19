import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_

class HierarchicalConcatFusion(nn.Module):
    """
    🎯 계층적 게이팅 기반 Concat Fusion
    
    FusionConcat을 기반으로 한 간단한 계층적 게이팅 구현
    
    핵심 아이디어:
    1. RGB의 불확실성(엔트로피) 측정 → master_weight 생성
    2. RGB 확실할 때: master_weight ≈ 0 → Motion 센서 차단
    3. RGB 불확실할 때: master_weight ≈ 1 → Motion 센서 활용
    4. FusionConcat과 동일한 구조: [RGB, w*Gyro, w*Acce] → 3072 → 512
    
    직관: "주력 선수(RGB)가 흔들릴 때만 교체 선수(Motion 센서)를 투입한다"
    """
    
    def __init__(self, feature_dim, modality, dropout, alpha=5.0, entropy_dim=256):
        """
        Args:
            feature_dim: 각 모달리티 특징 차원 (1024)
            modality: 모달리티 리스트 ["RGB", "Gyro", "Acce"]
            dropout: 드롭아웃 확률
            alpha: RGB 엔트로피 게이트 민감도
            entropy_dim: RGB 엔트로피 계산용 투영 차원
        """
        super().__init__()
        self.modality = modality
        self.dropout = dropout
        self.alpha = alpha
        self.entropy_dim = entropy_dim
        
        # 모달리티 인덱스 매핑
        self.modality_to_idx = {m: i for i, m in enumerate(self.modality)}
        
        # RGB 엔트로피 계산용 투영 레이어 (간단하게)
        self.rgb_entropy_proj = nn.Linear(feature_dim, entropy_dim)
        normal_(self.rgb_entropy_proj.weight, 0, 0.001)
        constant_(self.rgb_entropy_proj.bias, 0)
        
        # FusionConcat과 동일한 구조
        if len(self.modality) > 1:  # Multi-modal fusion
            input_dim = len(self.modality) * feature_dim  # 3 * 1024 = 3072
            self.fc1 = nn.Linear(input_dim, 512)
            self.relu = nn.ReLU()
            
            # weight init (FusionConcat과 동일)
            normal_(self.fc1.weight, 0, 0.001)
            constant_(self.fc1.bias, 0)
        
        # Dropout layer (FusionConcat과 동일)
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # 디버깅 플래그
        self._debug_printed = False

    def _compute_rgb_uncertainty(self, f_rgb):
        """
        RGB 불확실성 계산
        
        Args:
            f_rgb: RGB 특징 벡터 [Batch, feature_dim]
            
        Returns:
            master_weight: [Batch] 마스터 가중치 (0~1)
            rgb_entropy: [Batch] RGB 엔트로피 (로깅용)
        """
        # RGB 특징을 작은 차원으로 투영
        rgb_proj = self.rgb_entropy_proj(f_rgb)  # [Batch, entropy_dim]
        
        # 확률 분포로 변환
        rgb_prob = F.softmax(rgb_proj, dim=1)  # [Batch, entropy_dim]
        
        # 엔트로피 계산: H(p) = -sum(p * log(p))
        eps = 1e-8
        rgb_entropy = -torch.sum(rgb_prob * torch.log(rgb_prob + eps), dim=1)  # [Batch]
        
        # 엔트로피 정규화 (0~1 범위)
        max_entropy = torch.log(torch.tensor(self.entropy_dim, device=rgb_entropy.device))
        rgb_entropy_norm = rgb_entropy / max_entropy  # [Batch]
        
        # 마스터 가중치: 엔트로피 높을수록 Motion 센서 필요
        master_weight = torch.sigmoid(self.alpha * rgb_entropy_norm)  # [Batch]
        
        return master_weight, rgb_entropy_norm

    def forward(self, inputs, targets=None):
        """
        Forward pass: FusionConcat + 계층적 게이팅
        
        Args:
            inputs: List[Tensor] - [f_rgb, f_gyro, f_acce]
            
        Returns:
            dict: 융합된 특징 및 게이팅 정보
        """
        if len(self.modality) > 1:  # Multi-modal fusion
            # 각 모달리티 특징 분리
            f_rgb = inputs[self.modality_to_idx['RGB']]
            f_gyro = inputs[self.modality_to_idx['Gyro']] if 'Gyro' in self.modality_to_idx else None
            f_acce = inputs[self.modality_to_idx['Acce']] if 'Acce' in self.modality_to_idx else None
            
            # 🎯 RGB 불확실성 기반 마스터 가중치 계산
            master_weight, rgb_entropy = self._compute_rgb_uncertainty(f_rgb)
            
            # Motion 센서에 마스터 가중치 적용
            weighted_inputs = [f_rgb]  # RGB는 항상 그대로
            
            if f_gyro is not None:
                # master_weight를 [Batch, 1]로 확장하여 브로드캐스팅
                w_gyro = master_weight.unsqueeze(1)  # [Batch, 1]
                weighted_gyro = w_gyro * f_gyro  # [Batch, feature_dim]
                weighted_inputs.append(weighted_gyro)
            
            if f_acce is not None:
                w_acce = master_weight.unsqueeze(1)  # [Batch, 1]
                weighted_acce = w_acce * f_acce  # [Batch, feature_dim]
                weighted_inputs.append(weighted_acce)
            
            # FusionConcat과 동일한 방식으로 결합
            x = torch.cat(weighted_inputs, dim=1)  # [Batch, 3072]
            x = self.fc1(x)  # [Batch, 512]
            x = self.relu(x)
            x = self.dropout_layer(x)
            
        else:  # Single modality - FusionConcat과 동일
            x = inputs[0]
            x = self.dropout_layer(x)
            master_weight = torch.ones(x.size(0), device=x.device)  # 더미 값
            rgb_entropy = torch.zeros(x.size(0), device=x.device)   # 더미 값
            
        # 디버깅 정보 출력 (첫 번째 forward에서만)
        if not self._debug_printed:
            print(f"🎯 HierarchicalConcatFusion Debug:")
            print(f"   Modality count: {len(self.modality)}")
            print(f"   Input shapes: {[inp.shape for inp in inputs]}")
            print(f"   RGB entropy range: [{rgb_entropy.min().item():.3f}, {rgb_entropy.max().item():.3f}]")
            print(f"   Master weight range: [{master_weight.min().item():.3f}, {master_weight.max().item():.3f}]")
            print(f"   Output shape: {x.shape}")
            print(f"   Alpha parameter: {self.alpha}")
            print(f"   Architecture: RGB_entropy → master_weight → Motion_gating")
            print(f"   Structure: [RGB_1024, w*Gyro_1024, w*Acce_1024] → 3072 → 512")
            self._debug_printed = True
            
        return {
            'features': x,
            'master_weight': master_weight.detach(),  # 마스터 가중치 (로깅용)
            'rgb_entropy': rgb_entropy.detach(),      # RGB 엔트로피 (로깅용)
        }
