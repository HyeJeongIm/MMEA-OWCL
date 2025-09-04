import torch
import torch.nn.functional as F
import numpy as np
import logging
from .base_ood import BaseOODDetector
from .energy import EnergyDetector

class LTSRGBOnlyNoNormDetector(BaseOODDetector):
    """LTS RGB Only WITHOUT L2 normalization - for debugging"""
    
    def __init__(self, model, device='cuda', temperature=1.0, percentile=65):
        super().__init__(model, device)
        self.temperature = temperature
        self.percentile = percentile
        
        # Energy detector를 내부적으로 사용
        self.energy_detector = EnergyDetector(model, device, temperature)
        
    def lts_scale(self, x, percentile=None):
        """
        Compute LTS scale for RGB features only
        """
        if percentile is None:
            percentile = self.percentile
            
        assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
        assert 0 <= percentile <= 100, f"Percentile must be in [0, 100], got {percentile}"
        
        logging.info("  ⚡ Computing LTS scale (RGB Only, No Norm):")
        
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
        avg_scale = scale.mean().item()
        
        logging.info(f"    📈 Percentile: {percentile}%, Selected features: {k}/{n}")
        logging.info(f"    📊 S1 (all features sum): {avg_s1:.4f}")
        logging.info(f"    📊 S2 (top-k features sum): {avg_s2:.4f}")
        logging.info(f"    🎯 LTS Scale - Avg: {avg_scale:.4f}, Min: {scale.min().item():.4f}, Max: {scale.max().item():.4f}")
        
        return scale
    
    def _compute_scores_from_logits(self, logits):
        """
        Fallback: use regular energy score when features not available
        """
        return self.energy_detector._compute_scores_from_logits(logits)
    
    def compute_scores_with_features(self, logits, individual_features):
        """
        Main method: Compute LTS scores using RGB modality only (NO L2 normalization)
        Args:
            logits: (batch_size, num_classes) - pre-computed logits
            individual_features: list of feature tensors for each modality
                                [RGB, Gyro, Acce, ...] - RGB is at index 0
        Returns:
            scores: numpy array of shape (batch_size,)
        """
        logging.info("🚀 [LTS_RGB_Only_No_Norm] Starting computation with RGB features (NO L2 normalization)")
        
        # Ensure all tensors are on the same device
        device = logits.device
        
        # Extract RGB features only (assuming RGB is the first modality)
        if len(individual_features) == 0:
            logging.error("No individual features provided!")
            return self._compute_scores_from_logits(logits)
        
        rgb_features = individual_features[0].to(device)  # RGB is index 0
        
        # 입력 데이터 정보 로깅
        logging.info(f"  📥 Input logits shape: {logits.shape}")
        logging.info(f"  📥 RGB features shape: {rgb_features.shape}")
        logging.info(f"  🎯 Using RGB modality only (index 0 out of {len(individual_features)} modalities)")
        logging.info(f"  ⚠️ NO L2 normalization applied!")
        
        # RGB features 통계 로깅 (정규화 없음)
        rgb_magnitude = torch.mean(torch.abs(rgb_features)).item()
        rgb_std = rgb_features.std().item()
        rgb_mean = rgb_features.mean().item()
        
        logging.info(f"  ✨ RGB feature magnitude: {rgb_magnitude:.4f}")
        logging.info(f"  ✨ RGB feature mean: {rgb_mean:.4f}")
        logging.info(f"  ✨ RGB feature std: {rgb_std:.4f}")
        
        # Step 2: Compute LTS scale from RGB features (NO normalization)
        lts_scale = self.lts_scale(rgb_features)
        
        # 🔍 DEBUG: LTS_RGB_Only_No_Norm scale statistics
        logging.info(f"  🔍 [DEBUG] LTS_RGB_Only_No_Norm scale - Mean: {lts_scale.mean().item():.6f}, Std: {lts_scale.std().item():.6f}")
        logging.info(f"  🔍 [DEBUG] RGB features raw - Mean: {rgb_features.mean().item():.6f}, Std: {rgb_features.std().item():.6f}")
        
        # Step 3: Apply LTS scale to logits
        scaled_logits = logits * lts_scale
        
        # Scaled logits 통계 로깅
        original_logits_mean = logits.mean().item()
        scaled_logits_mean = scaled_logits.mean().item()
        scaling_factor = scaled_logits_mean / (original_logits_mean + 1e-8)
        
        logging.info(f"  📊 Original logits mean: {original_logits_mean:.4f}")
        logging.info(f"  📊 Scaled logits mean: {scaled_logits_mean:.4f}")
        logging.info(f"  📊 Effective scaling factor: {scaling_factor:.4f}")
        
        # Step 4: Use Energy-based method to compute final OOD scores
        final_scores = self.energy_detector._compute_scores_from_logits(scaled_logits)
        
        # 최종 점수 통계 로깅
        avg_score = np.mean(final_scores)
        min_score = np.min(final_scores)
        max_score = np.max(final_scores)
        
        logging.info(f"  🎯 Final OOD scores (RGB Only, No Norm) - Avg: {avg_score:.4f}, Min: {min_score:.4f}, Max: {max_score:.4f}")
        logging.info("✅ [LTS_RGB_Only_No_Norm] Computation completed")
        
        return final_scores
