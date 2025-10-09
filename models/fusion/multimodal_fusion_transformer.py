import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
import math

class MultiModalFusionTransformer(nn.Module):
    """
    🎯 Multi-Modal Fusion Transformer (초소형 퓨전 트랜스포머)
    
    핵심 아이디어:
    각 모달리티의 특징을 하나의 '토큰(Token)'으로 간주하고, 
    초소형 트랜스포머 인코더에 입력하여 모달리티 간의 모든 상호작용을 학습합니다.
    
    동작 방식:
    1. 각 모달리티 특징을 d_model 차원으로 projection
    2. 특별한 분류 토큰 [CLS] 추가
    3. 모든 토큰을 트랜스포머 인코더에 입력
    4. [CLS] 토큰의 출력을 최종 융합 특징으로 사용
    
    장점:
    - Self-Attention으로 모든 모달리티 쌍의 관계를 동시에 학습
    - 데이터에 따라 중요한 상호작용을 동적으로 파악
    - 새로운 모달리티 추가 시 확장성 우수
    """
    
    def __init__(self, feature_dim, modality, dropout, d_model=512, nhead=8, num_layers=2, dim_feedforward=1024):
        """
        Args:
            feature_dim: 각 모달리티 특징 차원 (1024)
            modality: 모달리티 리스트 ["RGB", "Gyro", "Acce"]
            dropout: 드롭아웃 확률
            d_model: 트랜스포머 모델 차원 (기본값: 512)
            nhead: 멀티헤드 어텐션 헤드 수 (기본값: 8)
            num_layers: 트랜스포머 인코더 레이어 수 (기본값: 2)
            dim_feedforward: FFN 차원 (기본값: 1024)
        """
        super().__init__()
        self.modality = modality
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 모달리티 인덱스 매핑
        self.modality_to_idx = {m: i for i, m in enumerate(self.modality)}
        self.num_modalities = len(self.modality)
        
        # 🎯 각 모달리티를 d_model 차원으로 projection
        self.modality_projections = nn.ModuleDict()
        for modality_name in self.modality:
            self.modality_projections[modality_name] = nn.Linear(feature_dim, d_model)
            # 가중치 초기화
            normal_(self.modality_projections[modality_name].weight, 0, 0.02)
            constant_(self.modality_projections[modality_name].bias, 0)
        
        # 🎯 특별한 분류 토큰 [CLS]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        normal_(self.cls_token, 0, 0.02)
        
        # 🎯 위치 인코딩 (선택적 - 모달리티 순서 정보)
        self.use_positional_encoding = True
        if self.use_positional_encoding:
            max_seq_len = self.num_modalities + 1  # +1 for CLS token
            self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
            normal_(self.positional_encoding, 0, 0.02)
        
        # 🎯 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # [Batch, Seq, Feature] 순서
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # 🎯 최종 출력 정규화
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 디버깅 및 분석용
        self.first_forward = True
        self.attention_weights = None  # 어텐션 가중치 저장용

    def _pick_features(self, features):
        """features 리스트에서 각 모달리티 특징 추출"""
        f_rgb = features[self.modality_to_idx['RGB']] if 'RGB' in self.modality_to_idx else None
        f_gyro = features[self.modality_to_idx['Gyro']] if 'Gyro' in self.modality_to_idx else None
        f_acce = features[self.modality_to_idx['Acce']] if 'Acce' in self.modality_to_idx else None
        return f_rgb, f_gyro, f_acce

    def _create_tokens(self, features):
        """
        모달리티 특징들을 트랜스포머 토큰으로 변환
        
        Args:
            features: List[Tensor] - [f_rgb, f_gyro, f_acce]
            
        Returns:
            tokens: [Batch, num_tokens, d_model] - [CLS] + modality tokens
            token_names: List[str] - 토큰 이름 리스트
        """
        # 각 모달리티 특징 분리
        f_rgb, f_gyro, f_acce = self._pick_features(features)
        modality_features = {'RGB': f_rgb, 'Gyro': f_gyro, 'Acce': f_acce}
        
        batch_size = features[0].size(0)
        tokens = []
        token_names = ['CLS']
        
        # 🎯 [CLS] 토큰 추가
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [Batch, 1, d_model]
        tokens.append(cls_tokens)
        
        # 🎯 각 모달리티 특징을 토큰으로 변환
        for modality_name in self.modality:
            feature = modality_features[modality_name]
            if feature is not None:
                # Feature projection: [Batch, feature_dim] → [Batch, d_model]
                projected_feature = self.modality_projections[modality_name](feature)
                # Add sequence dimension: [Batch, d_model] → [Batch, 1, d_model]
                projected_feature = projected_feature.unsqueeze(1)
                tokens.append(projected_feature)
                token_names.append(modality_name)
        
        # 모든 토큰 결합: [Batch, num_tokens, d_model]
        tokens = torch.cat(tokens, dim=1)
        
        # 🎯 위치 인코딩 추가 (선택적)
        if self.use_positional_encoding:
            seq_len = tokens.size(1)
            tokens = tokens + self.positional_encoding[:, :seq_len, :]
        
        return tokens, token_names

    def _extract_attention_weights(self):
        """
        트랜스포머 인코더에서 어텐션 가중치 추출
        (디버깅 및 분석용)
        """
        attention_weights = []
        
        # 각 레이어의 어텐션 가중치 수집
        for layer in self.transformer_encoder.layers:
            if hasattr(layer.self_attn, 'attention_weights'):
                attention_weights.append(layer.self_attn.attention_weights)
        
        return attention_weights

    def forward(self, features):
        """
        Forward pass: Multi-Modal Fusion Transformer
        
        Args:
            features: List[Tensor] - [f_rgb, f_gyro, f_acce]
            
        Returns:
            dict: 융합된 특징 + 어텐션 정보
        """
        # 🎯 모달리티 특징들을 트랜스포머 토큰으로 변환
        tokens, token_names = self._create_tokens(features)  # [Batch, num_tokens, d_model]
        
        # 🎯 트랜스포머 인코더 통과
        # Self-Attention으로 모든 토큰 간의 상호작용 학습
        encoded_tokens = self.transformer_encoder(tokens)  # [Batch, num_tokens, d_model]
        
        # 🎯 [CLS] 토큰의 출력을 최종 융합 특징으로 사용
        cls_output = encoded_tokens[:, 0, :]  # [Batch, d_model]
        cls_output = self.layer_norm(cls_output)
        
        # 🎯 어텐션 가중치 분석 (첫 번째 레이어의 첫 번째 헤드)
        attention_analysis = self._analyze_attention_patterns(encoded_tokens, token_names)
        
        # 디버깅 정보 출력 (첫 번째 forward에서만)
        if self.first_forward:
            print(f"🎯 MultiModalFusionTransformer Debug:")
            print(f"   Available tokens: {token_names}")
            print(f"   Token sequence length: {tokens.size(1)}")
            print(f"   Transformer config: d_model={self.d_model}, nhead={self.nhead}, layers={self.num_layers}")
            print(f"   CLS output range: [{cls_output.min().item():.3f}, {cls_output.max().item():.3f}]")
            print(f"   Architecture: modality_projection → tokens → transformer_encoder → cls_token")
            self.first_forward = False
        
        return {
            'features': cls_output,
            'tokens': encoded_tokens,
            'token_names': token_names,
            'attention_analysis': attention_analysis,
            'num_tokens': len(token_names)
        }

    def _analyze_attention_patterns(self, encoded_tokens, token_names):
        """
        어텐션 패턴 분석 (간단한 토큰 간 유사도 기반)
        
        Args:
            encoded_tokens: [Batch, num_tokens, d_model]
            token_names: List[str]
            
        Returns:
            dict: 어텐션 분석 결과
        """
        batch_size, num_tokens, d_model = encoded_tokens.shape
        
        # 토큰 간 코사인 유사도 계산
        # [Batch, num_tokens, d_model] → [Batch, num_tokens, num_tokens]
        normalized_tokens = F.normalize(encoded_tokens, p=2, dim=-1)
        similarity_matrix = torch.bmm(normalized_tokens, normalized_tokens.transpose(1, 2))
        
        # 배치 평균 계산
        avg_similarity = similarity_matrix.mean(dim=0)  # [num_tokens, num_tokens]
        
        # CLS 토큰과 다른 토큰들 간의 관계 분석
        cls_similarities = avg_similarity[0, 1:]  # CLS와 다른 토큰들 간 유사도
        modality_names = token_names[1:]  # CLS 제외
        
        # 모달리티 간 상호작용 강도
        modality_interactions = {}
        for i, name_i in enumerate(modality_names):
            for j, name_j in enumerate(modality_names):
                if i < j:  # 중복 제거
                    interaction_key = f"{name_i}-{name_j}"
                    modality_interactions[interaction_key] = avg_similarity[i+1, j+1].item()
        
        return {
            'similarity_matrix': avg_similarity.detach(),
            'cls_modality_similarities': dict(zip(modality_names, cls_similarities.detach().tolist())),
            'modality_interactions': modality_interactions,
            'token_names': token_names
        }

    def get_attention_maps(self, features):
        """
        실제 어텐션 맵을 추출하는 함수 (고급 분석용)
        
        Args:
            features: List[Tensor] - [f_rgb, f_gyro, f_acce]
            
        Returns:
            dict: 각 레이어별 어텐션 맵
        """
        # Hook을 사용하여 실제 어텐션 가중치 추출
        attention_maps = {}
        
        def hook_fn(module, input, output):
            # MultiheadAttention의 출력에서 어텐션 가중치 추출
            if hasattr(module, 'attention_weights'):
                attention_maps[f'layer_{len(attention_maps)}'] = module.attention_weights
        
        # Hook 등록
        hooks = []
        for i, layer in enumerate(self.transformer_encoder.layers):
            hook = layer.self_attn.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            result = self.forward(features)
        
        # Hook 제거
        for hook in hooks:
            hook.remove()
        
        return {
            'attention_maps': attention_maps,
            'result': result
        }
