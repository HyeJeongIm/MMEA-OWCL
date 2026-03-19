import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_

class CrossAttentionFusion(nn.Module):
    """
    🎯 Cross-Attention 기반 동적 퓨전 모듈 (v1)
    
    핵심 컨셉:
    IMU(Gyro+Acce)를 '쿼리(Query)'로 사용하여 RGB 특징 중에서 현재 움직임과 
    가장 관련성이 높은 시각적 정보를 동적으로 선택하고 강조합니다.
    
    가설:
    "움직임(IMU) 정보는 시각(RGB) 정보의 모호성을 해결하는 결정적 맥락을 제공한다"
    
    동작 방식:
    1. Motion(Gyro+Acce) → Query: "현재 이런 움직임이 있는데..."
    2. RGB → Key/Value: "시각적 정보 중에서 관련있는 것은?"
    3. Cross-Attention: "움직임 맥락에서 중요한 시각 정보 추출"
    4. Final Fusion: "원본 Motion + 재해석된 RGB" 결합
    
    예시:
    - '손을 빠르게 흔드는' IMU 신호 → RGB에서 '손' 관련 시각 요소에 높은 가중치
    - '걷는' IMU 패턴 → RGB에서 '다리/발' 관련 시각 정보에 집중
    """
    
    def __init__(self, feature_dim, modality, dropout, embed_dim=512, num_heads=8):
        """
        Args:
            feature_dim (int): 각 모달리티 특징 차원 (1024)
            modality (list): 모달리티 리스트 ["RGB", "Gyro", "Acce"]
            dropout (float): 드롭아웃 확률
            embed_dim (int): Cross-Attention 내부 임베딩 차원 (기본값: 512)
            num_heads (int): 멀티헤드 어텐션의 헤드 수 (기본값: 8)
        """
        super().__init__()
        self.modality = modality
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # 모달리티 인덱스 매핑
        self.modality_to_idx = {m: i for i, m in enumerate(self.modality)}
        
        # 필수 모달리티 확인
        assert 'RGB' in self.modality_to_idx, "CrossAttentionFusion requires RGB modality"
        assert 'Gyro' in self.modality_to_idx, "CrossAttentionFusion requires Gyro modality"
        assert 'Acce' in self.modality_to_idx, "CrossAttentionFusion requires Acce modality"
        
        # 🎯 3-1. 역할 정의 및 투영 (Role Definition & Projection)
        
        # Motion(Gyro+Acce) → Query 투영
        # "현재 이런 움직임이 감지되었는데, 어떤 시각 정보가 중요할까?"
        self.motion_to_query = nn.Linear(feature_dim * 2, embed_dim)  # 2048 → 512
        
        # RGB → Key 투영  
        # "시각 정보의 각 부분이 어떤 특성을 가지는지 알려주는 키"
        self.rgb_to_key = nn.Linear(feature_dim, embed_dim)  # 1024 → 512
        
        # RGB → Value 투영
        # "실제로 전달할 시각 정보의 내용"
        self.rgb_to_value = nn.Linear(feature_dim, embed_dim)  # 1024 → 512
        
        # 🎯 3-2. Cross-Attention 연산 모듈
        # IMU Query가 RGB Key/Value와 상호작용하여 관련성 높은 정보 추출
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # [Batch, Seq, Feature] 순서
        )
        
        # 🎯 3-3. 최종 융합 레이어
        # 원본 Motion 특징(2048) + 재해석된 RGB 특징(512) = 2560 → 512
        self.final_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2 + embed_dim, 512),  # 2560 → 512
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        
        # 가중치 초기화
        self._init_weights()
        
        # 디버깅 플래그
        self.first_forward = True

    def _init_weights(self):
        """보수적 가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                normal_(module.weight, 0, 0.001)  # 작은 가중치로 시작
                if module.bias is not None:
                    constant_(module.bias, 0)

    def _pick_features(self, features):
        """features 리스트에서 각 모달리티 특징 추출"""
        f_rgb = features[self.modality_to_idx['RGB']]
        f_gyro = features[self.modality_to_idx['Gyro']]
        f_acce = features[self.modality_to_idx['Acce']]
        return f_rgb, f_gyro, f_acce

    def forward(self, features, targets=None):
        """
        Cross-Attention 기반 동적 퓨전
        
        Args:
            features: List[Tensor] - [f_rgb, f_gyro, f_acce]
            targets: 정답 레이블 (현재 사용 안 함)
            
        Returns:
            dict: 융합된 특징 + Cross-Attention 정보
        """
        # 각 모달리티 특징 분리
        f_rgb, f_gyro, f_acce = self._pick_features(features)
        batch_size = f_rgb.size(0)
        
        # 🎯 3-1. 역할 정의 및 투영 (Role Definition & Projection)
        
        # Motion 특징 생성: Gyro + Acce 결합
        # "현재 감지된 움직임의 전체적인 패턴"
        feat_motion = torch.cat([f_gyro, f_acce], dim=1)  # [Batch, 2048]
        
        # Sequence 차원 추가 (MultiheadAttention 요구사항)
        feat_motion = feat_motion.unsqueeze(1)  # [Batch, 1, 2048]
        f_rgb_seq = f_rgb.unsqueeze(1)          # [Batch, 1, 1024]
        
        # Query: Motion → "이런 움직임에서 중요한 시각 정보는?"
        query_motion = self.motion_to_query(feat_motion)  # [Batch, 1, 512]
        
        # Key: RGB → "시각 정보의 각 특성"
        key_rgb = self.rgb_to_key(f_rgb_seq)  # [Batch, 1, 512]
        
        # Value: RGB → "실제 시각 정보 내용"
        value_rgb = self.rgb_to_value(f_rgb_seq)  # [Batch, 1, 512]
        
        # 🎯 3-2. Cross-Attention 연산
        # "Motion 맥락에서 RGB 정보를 재해석"
        attn_output, attn_weights = self.cross_attention(
            query=query_motion,  # [Batch, 1, 512] - Motion이 묻는다
            key=key_rgb,         # [Batch, 1, 512] - RGB가 답한다  
            value=value_rgb      # [Batch, 1, 512] - RGB 정보 전달
        )
        
        # Sequence 차원 제거
        attn_output = attn_output.squeeze(1)  # [Batch, 512]
        feat_motion = feat_motion.squeeze(1)  # [Batch, 2048]
        
        # 🎯 3-3. 최종 융합 (Final Fusion)
        # 원본 Motion 특징 + 재해석된 RGB 특징
        fused_features = torch.cat([feat_motion, attn_output], dim=1)  # [Batch, 2560]
        
        # 최종 출력 생성
        output = self.final_fusion(fused_features)  # [Batch, 512]
        
        # 디버깅 정보 출력 (첫 번째 forward에서만)
        if self.first_forward:
            print(f"🎯 CrossAttentionFusion Debug:")
            print(f"   Motion Features: {feat_motion.shape} (Gyro + Acce)")
            print(f"   RGB Features: {f_rgb.shape}")
            print(f"   Query (Motion): {query_motion.shape}")
            print(f"   Key/Value (RGB): {key_rgb.shape}")
            print(f"   Attention Output: {attn_output.shape}")
            print(f"   Final Fusion Input: {fused_features.shape}")
            print(f"   Final Output: {output.shape}")
            print(f"   Architecture: Motion(Query) × RGB(Key/Value) → Cross-Attention → Fusion")
            self.first_forward = False
        
        return {
            'features': output,
            'attention_weights': attn_weights.detach(),
            'motion_features': feat_motion.detach(),
            'reinterpreted_rgb': attn_output.detach(),
            'fusion_type': 'cross_attention'
        }

    def get_attention_analysis(self, features):
        """
        Cross-Attention 패턴 분석 (디버깅/시각화용)
        
        Returns:
            dict: Attention 가중치 및 분석 정보
        """
        with torch.no_grad():
            result = self.forward(features)
            
            attention_weights = result['attention_weights']  # [Batch, num_heads, 1, 1]
            motion_features = result['motion_features']      # [Batch, 2048]
            reinterpreted_rgb = result['reinterpreted_rgb']  # [Batch, 512]
            
            # Attention 강도 분석
            avg_attention = attention_weights.mean(dim=1).squeeze()  # [Batch]
            
            # Motion과 재해석된 RGB 간의 유사도 계산을 위해 차원 맞추기
            # Motion 특징을 512차원으로 projection
            motion_projected = self.motion_to_query(motion_features.unsqueeze(1)).squeeze(1)  # [Batch, 512]
            
            motion_rgb_similarity = F.cosine_similarity(
                motion_projected, 
                reinterpreted_rgb, 
                dim=1
            )
            
            return {
                'attention_strength': avg_attention.mean().item(),
                'motion_rgb_similarity': motion_rgb_similarity.mean().item(),
                'attention_weights': attention_weights,
                'interpretation': self._interpret_attention(avg_attention.mean().item())
            }
    
    def _interpret_attention(self, attention_strength):
        """Attention 강도 해석"""
        if attention_strength > 0.8:
            return "Strong Motion-RGB correlation detected"
        elif attention_strength > 0.5:
            return "Moderate Motion-RGB interaction"
        else:
            return "Weak Motion-RGB relationship"
