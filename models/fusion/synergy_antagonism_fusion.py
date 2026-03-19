import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_

class SynergyAntagonismFusion(nn.Module):
    """
    🎯 Synergy-Antagonism Fusion (시너지-길항 융합)
    
    핵심 아이디어:
    IMU 센서(Gyro, Acce)는 서로 밀접한 관련이 있습니다. 
    두 센서가 '비슷하게' 움직일 때(시너지)와 '다르게' 움직일 때(길항)의 의미가 다릅니다.
    
    예시:
    - '걷기': 두 센서가 조화롭게 움직임 (시너지)
    - '넘어지기': 두 센서가 순간적으로 매우 다른 패턴 (길항)
    
    구조:
    1. 각 모달리티 특징 추출: f_rgb, f_gyro, f_acce
    2. IMU 상호작용 계산:
       - f_diff = f_gyro - f_acce (차이/길항)
       - f_mul = f_gyro * f_acce (유사성/시너지)
    3. 시너지 특징 생성: synergy_mlp([f_diff, f_mul]) → f_synergy
    4. 최종 융합: fc([f_rgb, f_gyro, f_acce, f_synergy]) → 512
    """
    
    def __init__(self, feature_dim, modality, dropout, synergy_dim=512):
        """
        Args:
            feature_dim: 각 모달리티 특징 차원 (1024)
            modality: 모달리티 리스트 ["RGB", "Gyro", "Acce"]
            dropout: 드롭아웃 확률
            synergy_dim: 시너지 특징 차원 (기본값: 512)
        """
        super().__init__()
        self.modality = modality
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.synergy_dim = synergy_dim
        
        # 모달리티 인덱스 매핑
        self.modality_to_idx = {m: i for i, m in enumerate(self.modality)}
        
        # IMU 센서가 모두 있는지 확인
        self.has_imu = 'Gyro' in self.modality_to_idx and 'Acce' in self.modality_to_idx
        
        if self.has_imu:
            # 🎯 시너지 모듈: IMU 상호작용 → 시너지 특징
            # 입력: [f_diff, f_mul] = [feature_dim * 2]
            # 출력: f_synergy = [synergy_dim]
            self.synergy_mlp = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),  # 2*1024 → 1024
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, synergy_dim),      # 1024 → 512
                nn.ReLU()
            )
            
            # 시너지 MLP 가중치 초기화
            for layer in self.synergy_mlp:
                if isinstance(layer, nn.Linear):
                    normal_(layer.weight, 0, 0.001)
                    constant_(layer.bias, 0)
            
            # 🎯 최종 융합 레이어
            # 입력: [f_rgb, f_gyro, f_acce, f_synergy] 
            # RGB(1024) + Gyro(1024) + Acce(1024) + Synergy(512) = 3584
            final_input_dim = len(self.modality) * feature_dim + synergy_dim
            self.final_fc = nn.Linear(final_input_dim, 512)
            
        else:
            # IMU 센서가 없는 경우: 기본 Concat 방식
            final_input_dim = len(self.modality) * feature_dim
            self.final_fc = nn.Linear(final_input_dim, 512)
        
        # 최종 FC 레이어 초기화
        normal_(self.final_fc.weight, 0, 0.001)
        constant_(self.final_fc.bias, 0)
        
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # 디버깅 플래그
        self.first_forward = True

    def _pick_features(self, features):
        """features 리스트에서 각 모달리티 특징 추출"""
        f_rgb = features[self.modality_to_idx['RGB']] if 'RGB' in self.modality_to_idx else None
        f_gyro = features[self.modality_to_idx['Gyro']] if 'Gyro' in self.modality_to_idx else None
        f_acce = features[self.modality_to_idx['Acce']] if 'Acce' in self.modality_to_idx else None
        return f_rgb, f_gyro, f_acce

    def _compute_synergy_features(self, f_gyro, f_acce):
        """
        IMU 센서 간 상호작용 특징 계산
        
        Args:
            f_gyro: [Batch, feature_dim] Gyroscope 특징
            f_acce: [Batch, feature_dim] Accelerometer 특징
            
        Returns:
            f_synergy: [Batch, synergy_dim] 시너지 특징
            interaction_info: dict - 디버깅용 상호작용 정보
        """
        # 🎯 길항 관계 (Antagonism): 두 센서의 차이
        # 큰 차이 → 갑작스러운 변화, 비정상적 움직임 (예: 넘어지기, 충격)
        f_diff = f_gyro - f_acce  # [Batch, feature_dim]
        
        # 🎯 시너지 관계 (Synergy): 두 센서의 상호작용
        # 높은 곱셈값 → 두 센서가 함께 활성화, 조화로운 움직임 (예: 걷기, 뛰기)
        f_mul = f_gyro * f_acce   # [Batch, feature_dim]
        
        # 두 상호작용 특징을 결합하여 시너지 MLP에 입력
        interaction_features = torch.cat([f_diff, f_mul], dim=1)  # [Batch, feature_dim * 2]
        f_synergy = self.synergy_mlp(interaction_features)        # [Batch, synergy_dim]
        
        # 디버깅용 정보 계산
        interaction_info = {
            'antagonism_magnitude': torch.mean(torch.abs(f_diff), dim=1),  # 길항 강도
            'synergy_magnitude': torch.mean(torch.abs(f_mul), dim=1),      # 시너지 강도
            'interaction_ratio': torch.mean(torch.abs(f_diff), dim=1) / (torch.mean(torch.abs(f_mul), dim=1) + 1e-8)
        }
        
        return f_synergy, interaction_info

    def forward(self, features):
        """
        Forward pass: Synergy-Antagonism Fusion
        
        Args:
            features: List[Tensor] - [f_rgb, f_gyro, f_acce]
            
        Returns:
            dict: 융합된 특징 + 시너지 정보
        """
        # 각 모달리티 특징 분리
        f_rgb, f_gyro, f_acce = self._pick_features(features)
        
        # 사용 가능한 특징들을 리스트로 수집
        available_features = []
        modality_names = []
        
        for modality_name, feature in [('RGB', f_rgb), ('Gyro', f_gyro), ('Acce', f_acce)]:
            if feature is not None:
                available_features.append(feature)
                modality_names.append(modality_name)
        
        interaction_info = {}
        f_synergy = None
        
        # 🎯 IMU 센서가 모두 있는 경우: 시너지-길항 융합 적용
        if self.has_imu and f_gyro is not None and f_acce is not None:
            f_synergy, interaction_info = self._compute_synergy_features(f_gyro, f_acce)
            
            # 모든 특징 결합: [f_rgb, f_gyro, f_acce, f_synergy]
            all_features = available_features + [f_synergy]
            x = torch.cat(all_features, dim=1)
            
        else:
            # IMU 센서가 없는 경우: 기본 Concat 방식
            x = torch.cat(available_features, dim=1)
        
        # 최종 FC 레이어 통과
        x = self.final_fc(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        
        # 디버깅 정보 출력 (첫 번째 forward에서만)
        if self.first_forward:
            print(f"🎯 SynergyAntagonismFusion Debug:")
            print(f"   Available modalities: {modality_names}")
            print(f"   Has IMU sensors: {self.has_imu}")
            print(f"   Synergy feature dim: {self.synergy_dim}")
            if f_synergy is not None:
                print(f"   Synergy feature range: [{f_synergy.min().item():.3f}, {f_synergy.max().item():.3f}]")
            if interaction_info:
                print(f"   Avg antagonism: {interaction_info['antagonism_magnitude'].mean().item():.3f}")
                print(f"   Avg synergy: {interaction_info['synergy_magnitude'].mean().item():.3f}")
                print(f"   Avg interaction ratio: {interaction_info['interaction_ratio'].mean().item():.3f}")
            print(f"   Final feature dim: {x.shape[1]}")
            print(f"   Architecture: IMU_interaction → synergy_mlp → concat_fusion")
            self.first_forward = False
        
        return {
            'features': x,
            'synergy_features': f_synergy,
            'interaction_info': interaction_info,
            'modality_names': modality_names,
            'has_synergy': f_synergy is not None
        }
