import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_

class AuxiliaryHeadFusion(nn.Module):
    """
    🎯 간단한 Auxiliary Head 기반 융합 모듈
    
    핵심 아이디어:
    1. 각 모달리티에 간단한 auxiliary head (Linear) 추가
    2. Auxiliary head의 예측 신뢰도를 기반으로 가중치 계산
    3. 신뢰도 높은 모달리티에 더 높은 가중치 부여
    
    구조: FusionConcat + 각 모달리티별 auxiliary head
    - RGB → aux_head_rgb → confidence_rgb
    - Gyro → aux_head_gyro → confidence_gyro  
    - Acce → aux_head_acce → confidence_acce
    - 신뢰도 기반 가중치 → weighted fusion
    """
    
    def __init__(self, feature_dim, modality, dropout, num_classes=32, confidence_method="max_prob"):
        """
        Args:
            feature_dim: 각 모달리티 특징 차원 (1024)
            modality: 모달리티 리스트 ["RGB", "Gyro", "Acce"]
            dropout: 드롭아웃 확률
            num_classes: 클래스 수 (auxiliary head 출력 차원)
            confidence_method: 신뢰도 계산 방법 ("entropy", "max_prob")
        """
        super().__init__()
        self.modality = modality
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.confidence_method = confidence_method
        
        # 모달리티 인덱스 매핑
        self.modality_to_idx = {m: i for i, m in enumerate(self.modality)}
        
        # 🎯 각 모달리티별 간단한 auxiliary head (Linear만)
        self.auxiliary_heads = nn.ModuleDict()
        for modality_name in self.modality:
            self.auxiliary_heads[modality_name] = nn.Linear(feature_dim, num_classes)
            # 가중치 초기화
            normal_(self.auxiliary_heads[modality_name].weight, 0, 0.001)
            constant_(self.auxiliary_heads[modality_name].bias, 0)
        
        # FusionConcat과 동일한 최종 FC 레이어
        if len(self.modality) > 1:
            input_dim = len(self.modality) * feature_dim
            self.fc1 = nn.Linear(input_dim, 512)
            self.relu = nn.ReLU()
            normal_(self.fc1.weight, 0, 0.001)
            constant_(self.fc1.bias, 0)
        
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # 디버깅 플래그
        self.first_forward = True

    def _pick_features(self, features):
        """features 리스트에서 각 모달리티 특징 추출 (동적)"""
        modality_features = {}
        for i, modality_name in enumerate(self.modality):
            if i < len(features):
                modality_features[modality_name] = features[i]
            else:
                modality_features[modality_name] = None
        return modality_features

    def _compute_confidence(self, logits):
        """
        Auxiliary head 예측 결과로부터 신뢰도 계산
        
        Args:
            logits: [Batch, num_classes] auxiliary head 출력
            
        Returns:
            confidence: [Batch] 신뢰도 점수 (0~1)
        """
        probs = F.softmax(logits, dim=1)  # [Batch, num_classes]
        
        if self.confidence_method == "entropy":
            # 엔트로피 기반: 낮은 엔트로피 = 높은 신뢰도
            eps = 1e-8
            entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)  # [Batch]
            max_entropy = torch.log(torch.tensor(self.num_classes, device=entropy.device))
            confidence = 1.0 - (entropy / max_entropy)  # 엔트로피 낮을수록 신뢰도 높음
            
        elif self.confidence_method == "max_prob":
            # 최대 확률 기반: 높은 최대 확률 = 높은 신뢰도
            confidence, _ = torch.max(probs, dim=1)  # [Batch]
            
        else:
            # 기본값: 균등 신뢰도
            confidence = torch.ones(logits.size(0), device=logits.device) / len(self.modality)
        
        return confidence

    def forward(self, features, targets=None):
        """
        Forward pass: Auxiliary head 기반 가중치 + FusionConcat
        
        Args:
            features: List[Tensor] - [f_rgb, f_gyro, f_acce]
            targets: 정답 레이블 (auxiliary head 학습용, 선택적)
            
        Returns:
            dict: 융합된 특징 + auxiliary 정보
        """
        # 🎯 모든 경우에 동일한 로직 적용 (Single/Multi 구분 없음)
        # 각 모달리티 특징 분리
        modality_features = self._pick_features(features)
        
        # 🎯 각 모달리티별 auxiliary head 예측 및 신뢰도 계산
        auxiliary_logits = {}
        confidences = {}
        
        for modality_name, feature in modality_features.items():
            if feature is not None and modality_name in self.auxiliary_heads:
                aux_logits = self.auxiliary_heads[modality_name](feature)
                auxiliary_logits[modality_name] = aux_logits
                confidences[modality_name] = self._compute_confidence(aux_logits)
        
        # 신뢰도 기반 가중치 정규화
        if confidences and len(confidences) > 1:
            # Multi-modal: 소프트맥스로 가중치 분배
            confidence_tensor = torch.stack(list(confidences.values()), dim=1)  # [Batch, num_modalities]
            weights = F.softmax(confidence_tensor * 5.0, dim=1)  # 온도 파라미터 5.0
            
            # 각 모달리티에 가중치 적용
            weighted_features = []
            weight_list = []
            
            for modality_name, feature in modality_features.items():
                if feature is not None and modality_name in confidences:
                    weight = weights[:, list(confidences.keys()).index(modality_name)].unsqueeze(1)  # [Batch, 1]
                    weighted_feature = weight * feature
                    weighted_features.append(weighted_feature)
                    weight_list.append(weight.squeeze())
            
            # 결합 및 FC 레이어
            if len(self.modality) > 1:
                x = torch.cat(weighted_features, dim=1)
                x = self.fc1(x)
                x = self.relu(x)
            else:
                x = weighted_features[0]  # Single modality
            x = self.dropout_layer(x)
            
        elif confidences and len(confidences) == 1:
            # Single-modal with auxiliary head: 신뢰도를 가중치로 사용
            modality_name = list(confidences.keys())[0]
            feature = modality_features[modality_name]
            confidence = confidences[modality_name]
            
            # 🎯 신뢰도를 직접 가중치로 사용 (0~1 범위)
            # 높은 신뢰도 → 강한 특징, 낮은 신뢰도 → 약한 특징
            weight = confidence.unsqueeze(1)  # [Batch, 1]
            weighted_feature = weight * feature  # 신뢰도 기반 가중치 적용
            
            x = self.dropout_layer(weighted_feature)
            weight_list = [confidence]  # 실제 신뢰도 값 저장
            
        else:
            # Fallback: Auxiliary head 없거나 실패한 경우
            available_features = [f for f in modality_features.values() if f is not None]
            if len(available_features) > 1:
                x = torch.cat(available_features, dim=1)
                x = self.fc1(x)
                x = self.relu(x)
            else:
                x = available_features[0]
            x = self.dropout_layer(x)
            weight_list = [torch.ones(x.size(0), device=x.device) / len(available_features)] * len(available_features)
        
        # 디버깅 정보 출력 (첫 번째 forward에서만)
        if self.first_forward:
            print(f"🎯 AuxiliaryHeadFusion Debug:")
            print(f"   Modality count: {len(self.modality)}")
            print(f"   Confidence method: {self.confidence_method}")
            print(f"   Auxiliary heads: {list(self.auxiliary_heads.keys())}")
            if len(weight_list) > 0:
                print(f"   Weight ranges: [{weight_list[0].min().item():.3f}, {weight_list[0].max().item():.3f}]")
            print(f"   Architecture: feature → aux_head → confidence → weighted_fusion")
            self.first_forward = False
        
        # # Auxiliary loss 계산 (학습 시)
        # auxiliary_loss = 0.0
        # if targets is not None and auxiliary_logits:
        #     for modality_name, aux_logits in auxiliary_logits.items():
        #         auxiliary_loss += F.cross_entropy(aux_logits, targets)
        #     auxiliary_loss /= len(auxiliary_logits)  # 평균
        
        return {
            'features': x,
            'auxiliary_logits': auxiliary_logits,
            # 'auxiliary_loss': auxiliary_loss,
            'modality_weights': torch.stack(weight_list, dim=1).detach() if weight_list else None,
            'confidences': confidences,
        }
