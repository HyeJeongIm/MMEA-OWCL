import torch
import torch.nn.functional as F
import numpy as np
import logging
from .base_ood import BaseOODDetector
from .energy import EnergyDetector

class LTSLateFusionDetector(BaseOODDetector):
    """LTS Late Fusion: Each modality independently computes OOD scores, then combines them"""
    
    def __init__(self, model, device='cuda', temperature=1.0, percentile=65, fusion_method='average'):
        super().__init__(model, device)
        self.temperature = temperature
        self.percentile = percentile
        self.fusion_method = fusion_method  # 'average', 'weighted_average'
        
        # Energy detector를 내부적으로 사용
        self.energy_detector = EnergyDetector(model, device, temperature)
        
    def lts_scale(self, x, percentile=None):
        """
        Compute LTS scale for individual modality features
        """
        if percentile is None:
            percentile = self.percentile
            
        assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
        assert 0 <= percentile <= 100, f"Percentile must be in [0, 100], got {percentile}"
        
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
        
        return scale
    
    def compute_individual_modality_scores(self, logits, individual_features):
        """
        Compute OOD scores for each modality independently
        Args:
            logits: (batch_size, num_classes)
            individual_features: list of feature tensors for each modality
        Returns:
            modality_scores: list of numpy arrays, each (batch_size,)
            modality_weights: list of float weights for each modality
        """
        # 동적으로 모달리티 이름 생성
        common_modalities = ['RGB', 'Gyro', 'Acce', 'Flow', 'RGBDiff']
        num_modalities = len(individual_features)
        
        logging.info(f"  🔄 Computing independent OOD scores for {num_modalities} modalities:")
        
        modality_scores = []
        modality_weights = []
        
        for i, features in enumerate(individual_features):
            modality_name = common_modalities[i] if i < len(common_modalities) else f'Modality_{i}'
            
            # Ensure features are on correct device
            features = features.to(logits.device)
            
            # Apply L2 normalization to features
            features_normalized = F.normalize(features, p=2, dim=1)
            
            # Compute LTS scale for this modality
            lts_scale = self.lts_scale(features_normalized)
            
            # 🔍 DEBUG: Per-modality statistics
            logging.info(f"      🔍 [DEBUG] {modality_name} scale - Mean: {lts_scale.mean().item():.6f}, Std: {lts_scale.std().item():.6f}")
            logging.info(f"      🔍 [DEBUG] {modality_name} features - Mean: {features_normalized.mean().item():.6f}, Std: {features_normalized.std().item():.6f}")
            
            # Apply scale to logits
            scaled_logits = logits * lts_scale
            
            # Compute OOD scores using Energy method
            ood_scores = self.energy_detector._compute_scores_from_logits(scaled_logits)
            
            # Compute modality weight based on feature magnitude (representation norm)
            modality_weight = torch.mean(torch.abs(features)).item()
            
            modality_scores.append(ood_scores)
            modality_weights.append(modality_weight)
            
            # 각 모달리티별 통계 로깅
            avg_scale = lts_scale.mean().item()
            avg_score = np.mean(ood_scores)
            
            logging.info(f"    {modality_name}:")
            logging.info(f"      LTS Scale: {avg_scale:.4f}")
            logging.info(f"      OOD Score: {avg_score:.4f}")
            logging.info(f"      Weight (Rep. Norm): {modality_weight:.4f}")
        
        return modality_scores, modality_weights
    
    def combine_modality_scores(self, modality_scores, modality_weights):
        """
        Combine individual modality OOD scores
        Args:
            modality_scores: list of numpy arrays, each (batch_size,)
            modality_weights: list of float weights for each modality
        Returns:
            final_scores: numpy array (batch_size,)
        """
        # 동적으로 모달리티 이름 생성
        common_modalities = ['RGB', 'Gyro', 'Acce', 'Flow', 'RGBDiff']
        num_modalities = len(modality_scores)
        
        logging.info(f"  🔗 Combining {num_modalities} modality scores using '{self.fusion_method}' method:")
        
        if self.fusion_method == 'average':
            # Simple average
            final_scores = np.mean(modality_scores, axis=0)
            
            # 각 모달리티 동등한 가중치 로깅
            for i in range(num_modalities):
                modality_name = common_modalities[i] if i < len(common_modalities) else f'Modality_{i}'
                logging.info(f"    {modality_name}: Equal weight (33.3%)")
                
        elif self.fusion_method == 'weighted_average':
            # Weighted average based on representation norm
            weights = np.array(modality_weights)
            weights = weights / np.sum(weights)  # Normalize to sum to 1
            
            # Weighted combination
            final_scores = np.zeros_like(modality_scores[0])
            for i, (scores, weight) in enumerate(zip(modality_scores, weights)):
                final_scores += scores * weight
                
                # 각 모달리티 가중치 로깅
                modality_name = common_modalities[i] if i < len(common_modalities) else f'Modality_{i}'
                logging.info(f"    {modality_name}: Weight {weight:.3f} ({weight*100:.1f}%)")
        
        else:
            logging.error(f"Unknown fusion method: {self.fusion_method}")
            # Fallback to simple average
            final_scores = np.mean(modality_scores, axis=0)
        
        # 최종 점수 통계 로깅
        avg_final_score = np.mean(final_scores)
        min_final_score = np.min(final_scores)
        max_final_score = np.max(final_scores)
        
        logging.info(f"  🎯 Final combined scores - Avg: {avg_final_score:.4f}, Min: {min_final_score:.4f}, Max: {max_final_score:.4f}")
        
        return final_scores
    
    def _compute_scores_from_logits(self, logits):
        """
        Fallback: use regular energy score when features not available
        """
        return self.energy_detector._compute_scores_from_logits(logits)
    
    def compute_scores_with_features(self, logits, individual_features):
        """
        Main method: Compute Late Fusion LTS scores
        Args:
            logits: (batch_size, num_classes) - pre-computed logits
            individual_features: list of feature tensors for each modality
        Returns:
            scores: numpy array of shape (batch_size,)
        """
        logging.info("🚀 [LTS_Late_Fusion] Starting computation with late fusion approach")
        
        # Check if multiple modalities are available for meaningful late fusion
        if len(individual_features) < 2:
            logging.warning("❌ [LTS_Late_Fusion] Only 1 modality available - Late Fusion is not applicable.")
            logging.warning("    ❌ Returning None - this method requires multiple modalities.")
            return None
        
        # Ensure all tensors are on the same device
        device = logits.device
        individual_features = [feat.to(device) for feat in individual_features]
        
        # 입력 데이터 정보 로깅
        logging.info(f"  📥 Input logits shape: {logits.shape}")
        logging.info(f"  📥 Number of modalities: {len(individual_features)}")
        logging.info(f"  📥 Fusion method: {self.fusion_method}")
        for i, feat in enumerate(individual_features):
            logging.info(f"    Modality {i} features shape: {feat.shape}")
            # 🔍 각 모달리티 Raw 통계 확인
            feat_mean = feat.mean().item()
            feat_std = feat.std().item()
            logging.info(f"    Modality {i} raw stats: Mean={feat_mean:.6f}, Std={feat_std:.6f}")
        
        # Step 1: Compute independent OOD scores for each modality
        modality_scores, modality_weights = self.compute_individual_modality_scores(logits, individual_features)
        
        # Step 2: Combine modality scores
        final_scores = self.combine_modality_scores(modality_scores, modality_weights)
        
        logging.info("✅ [LTS_Late_Fusion] Computation completed")
        
        return final_scores
