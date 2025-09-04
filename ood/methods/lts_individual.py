import torch
import torch.nn.functional as F
import numpy as np
import logging
from .base_ood import BaseOODDetector
from .energy import EnergyDetector

class LTSIndividualDetector(BaseOODDetector):
    """LTS (Large-scale Temperature Scaling) using Individual Modality Features with Representation Norm Weighting"""
    
    def __init__(self, model, device='cuda', temperature=1.0, percentile=65):
        super().__init__(model, device)
        self.temperature = temperature
        self.percentile = percentile
        
        # Energy detector를 내부적으로 사용
        self.energy_detector = EnergyDetector(model, device, temperature)
        
    def compute_representation_norm_weights(self, individual_features):
        """
        Compute representation norm weights for each modality
        Args:
            individual_features: list of feature tensors for each modality
        Returns:
            weights: list of weight tensors, each (batch_size, 1)
        """
        norms = []
        # 동적으로 모달리티 이름 생성 (단일/멀티 모달리티 지원)
        common_modalities = ['RGB', 'Gyro', 'Acce', 'Flow', 'RGBDiff']
        num_modalities = len(individual_features)
        
        logging.info(f"🔍 [LTS_Individual] Computing representation norm weights for {num_modalities} modalities:")
        
        for i, features in enumerate(individual_features):
            # Compute representation norm: mean absolute value
            norm = torch.mean(torch.abs(features), dim=1, keepdim=True)  # (batch_size, 1)
            norms.append(norm)
            
            # 각 모달리티의 평균 norm 값 로깅
            avg_norm = norm.mean().item()
            modality_name = common_modalities[i] if i < len(common_modalities) else f'Modality_{i}'
            logging.info(f"  📊 {modality_name} - Avg Representation Norm: {avg_norm:.4f}")
        
        # 단일 모달리티인 경우 가중치는 1.0
        if num_modalities == 1:
            logging.info("  🎯 Single modality detected - Weight: 100.0%")
            return [torch.ones_like(norms[0])]
        
        # 멀티 모달리티인 경우 softmax로 정규화
        stacked_norms = torch.stack(norms, dim=0)  # (num_modalities, batch_size, 1)
        weights = torch.softmax(stacked_norms, dim=0)
        
        # 각 모달리티의 평균 가중치 (영향력) 로깅
        logging.info("  🎯 Modality Influence (Average Weights):")
        for i, weight in enumerate(weights):
            avg_weight = weight.mean().item() * 100  # 백분율로 변환
            modality_name = common_modalities[i] if i < len(common_modalities) else f'Modality_{i}'
            logging.info(f"    {modality_name}: {avg_weight:.1f}%")
        
        return [weights[i] for i in range(len(norms))]
    
    def combine_features_with_weights(self, individual_features, weights):
        """
        Combine individual modality features using representation norm weights (Improved version)
        Args:
            individual_features: list of feature tensors for each modality
            weights: list of weight tensors for each modality
        Returns:
            combined_features: (batch_size, feature_dim) - weighted average
        """
        weighted_features = []
        # 동적으로 모달리티 이름 생성 (단일/멀티 모달리티 지원)
        common_modalities = ['RGB', 'Gyro', 'Acce', 'Flow', 'RGBDiff']
        num_modalities = len(individual_features)
        
        logging.info(f"  🔗 Combining {num_modalities} modality features with weights:")
        
        for i, (features, weight) in enumerate(zip(individual_features, weights)):
            # Apply weight to each feature tensor
            weighted_feat = features * weight  # Broadcasting: (B, D) * (B, 1) = (B, D)
            weighted_features.append(weighted_feat)
            
            # 가중치 적용 후 특징의 평균 크기 로깅
            avg_magnitude = torch.mean(torch.abs(weighted_feat)).item()
            modality_name = common_modalities[i] if i < len(common_modalities) else f'Modality_{i}'
            logging.info(f"    {modality_name} weighted feature magnitude: {avg_magnitude:.4f}")
        
        # Sum all weighted features (weighted average since weights sum to 1)
        combined_features = sum(weighted_features)  # (batch_size, feature_dim)
        
        # 단일 모달리티인 경우 정규화 생략 (이미 원본 특징)
        if num_modalities == 1:
            logging.info("  ✨ Single modality - using original features without normalization")
        else:
            # 멀티 모달리티인 경우에만 정규화 적용
            before_norm_mean = combined_features.mean().item()
            before_norm_std = combined_features.std().item()
            combined_features = F.normalize(combined_features, p=2, dim=1)
            after_norm_mean = combined_features.mean().item()
            after_norm_std = combined_features.std().item()
            logging.info("  ✨ Multi-modality - applied L2 normalization to prevent R value increase")
            logging.info(f"  🔍 [DEBUG] Before norm: Mean={before_norm_mean:.6f}, Std={before_norm_std:.6f}")
            logging.info(f"  🔍 [DEBUG] After norm: Mean={after_norm_mean:.6f}, Std={after_norm_std:.6f}")
        
        # 결합된 특징의 통계 로깅
        combined_magnitude = torch.mean(torch.abs(combined_features)).item()
        combined_norm = torch.mean(torch.norm(combined_features, p=2, dim=1)).item()
        logging.info(f"  ✨ Combined feature magnitude: {combined_magnitude:.4f}")
        logging.info(f"  ✨ Combined feature L2 norm: {combined_norm:.4f}")
        
        return combined_features
        
    def lts_scale(self, x, percentile=None):
        """
        Compute LTS scale for combined features
        """
        if percentile is None:
            percentile = self.percentile
            
        assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
        assert 0 <= percentile <= 100, f"Percentile must be in [0, 100], got {percentile}"
        
        logging.info("  ⚡ Computing LTS scale:")
        
        # 1. Apply ReLU to remove negative values
        x = F.relu(x)
        
        # 2. Compute sum of all features
        s1 = x.sum(dim=1)  # (batch_size,)
        
        # 3. Select top-k features based on percentile
        n = x.shape[1]
        k = n - int(np.round(n * percentile / 100.0))
        
        if k <= 0:
            k = 1
            
        v, i = torch.topk(x, k, dim=1)
        
        # 4. Compute sum of selected features
        s2 = v.sum(dim=1)
        
        # 5. Compute scale: (s1 / s2)^2
        epsilon = 1e-8
        scale = s1 / (s2 + epsilon)
        scale = scale[:, None] ** 2
        
        # LTS scale 통계 로깅
        avg_s1 = s1.mean().item()
        avg_s2 = s2.mean().item()
        
        logging.info(f"    📈 Percentile: {percentile}%, Selected features: {k}/{n}")
        logging.info(f"    📊 S1 (all features sum): {avg_s1:.4f}")
        logging.info(f"    📊 S2 (top-k features sum): {avg_s2:.4f}")
        
        return scale
    
    def _compute_scores_from_logits(self, logits):
        """
        Fallback: use regular energy score when features not available
        """
        return self.energy_detector._compute_scores_from_logits(logits)
    
    def compute_scores_with_features(self, logits, individual_features):
        """
        Main method: Compute LTS scores using individual modality features
        Args:
            logits: (batch_size, num_classes) - pre-computed logits
            individual_features: list of feature tensors for each modality
                                Each tensor: (batch_size, feature_dim)
        Returns:
            scores: numpy array of shape (batch_size,)
        """
        logging.info("🚀 [LTS_Individual] Starting computation with features")
        
        # Ensure all tensors are on the same device
        device = logits.device
        individual_features = [feat.to(device) for feat in individual_features]
        
        # 입력 데이터 정보 로깅
        logging.info(f"  📥 Input logits shape: {logits.shape}")
        logging.info(f"  📥 Number of modalities: {len(individual_features)}")
        for i, feat in enumerate(individual_features):
            logging.info(f"    Modality {i} features shape: {feat.shape}")
            # 🔍 각 모달리티 통계 확인
            feat_mean = feat.mean().item()
            feat_std = feat.std().item()
            feat_min = feat.min().item()
            feat_max = feat.max().item()
            modality_name = ['RGB', 'Gyro', 'Acce'][i] if i < 3 else f'Modality_{i}'
            logging.info(f"    {modality_name} stats: Mean={feat_mean:.6f}, Std={feat_std:.6f}, Min={feat_min:.6f}, Max={feat_max:.6f}")
            
            # 🔍 특징 분포 확인 (첫 10개 값)
            first_10_vals = feat[0, :10].tolist()
            logging.info(f"    {modality_name} first 10 values: {[f'{v:.4f}' for v in first_10_vals]}")
        
        # Step 1: Compute representation norm weights for each modality
        weights = self.compute_representation_norm_weights(individual_features)
        
        # Step 2: Combine features using representation norm weights
        combined_features = self.combine_features_with_weights(individual_features, weights)
        
        # Step 3: Compute LTS scale from combined features
        lts_scale = self.lts_scale(combined_features)
        
        # 🔍 DEBUG: LTS_Individual scale statistics
        logging.info(f"  🔍 [DEBUG] LTS_Individual scale - Mean: {lts_scale.mean().item():.6f}, Std: {lts_scale.std().item():.6f}")
        logging.info(f"  🔍 [DEBUG] Combined features - Mean: {combined_features.mean().item():.6f}, Std: {combined_features.std().item():.6f}")
        
        # Step 4: Apply LTS scale to logits
        scaled_logits = logits * lts_scale
        
        # Scaled logits 통계 로깅
        original_logits_mean = logits.mean().item()
        scaled_logits_mean = scaled_logits.mean().item()
        scaling_factor = scaled_logits_mean / (original_logits_mean + 1e-8)
        
        logging.info(f"  📊 Original logits mean: {original_logits_mean:.4f}")
        logging.info(f"  📊 Scaled logits mean: {scaled_logits_mean:.4f}")
        logging.info(f"  📊 Effective scaling factor: {scaling_factor:.4f}")
        
        # Step 5: Use Energy-based method to compute final OOD scores
        final_scores = self.energy_detector._compute_scores_from_logits(scaled_logits)
        
        # 최종 점수 통계 로깅
        avg_score = np.mean(final_scores)
        min_score = np.min(final_scores)
        max_score = np.max(final_scores)
        
        logging.info(f"  🎯 Final OOD scores - Avg: {avg_score:.4f}, Min: {min_score:.4f}, Max: {max_score:.4f}")
        logging.info("✅ [LTS_Individual] Computation completed")
        
        return final_scores