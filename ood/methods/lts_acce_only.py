import torch
import torch.nn.functional as F
import numpy as np
import logging
from .base_ood import BaseOODDetector
from .energy import EnergyDetector

class LTSAcceOnlyDetector(BaseOODDetector):
    """LTS (Large-scale Temperature Scaling) using Accelerometer Modality Only"""
    
    def __init__(self, model, device='cuda', temperature=1.0, percentile=65):
        super().__init__(model, device)
        self.temperature = temperature
        self.percentile = percentile
        
        # Energy detector를 내부적으로 사용
        self.energy_detector = EnergyDetector(model, device, temperature)
        
    def lts_scale(self, x, percentile=None):
        """
        Compute LTS scale for Accelerometer features only
        """
        if percentile is None:
            percentile = self.percentile
            
        assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
        assert 0 <= percentile <= 100, f"Percentile must be in [0, 100], got {percentile}"
        
        logging.info("  ⚡ Computing LTS scale (Acce Only):")
        
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
        Main method: Compute LTS scores using Accelerometer modality only
        Args:
            logits: (batch_size, num_classes) - pre-computed logits
            individual_features: list of feature tensors for each modality
                                [RGB, Gyro, Acce, ...] - Acce is at index 2
        Returns:
            scores: numpy array of shape (batch_size,)
        """
        logging.info("🚀 [LTS_Acce_Only] Starting computation with Accelerometer features only")
        
        # Ensure all tensors are on the same device
        device = logits.device
        
        # Check if Accelerometer modality is available (need at least 3 modalities: RGB + Gyro + Acce)
        if len(individual_features) < 3:
            logging.warning("❌ [LTS_Acce_Only] Accelerometer modality not available! This model has fewer than 3 modalities.")
            logging.warning("    ❌ Returning None - this method is not applicable to models without Accelerometer.")
            return None
        
        # Extract Accelerometer features (index 2)
        acce_features = individual_features[2].to(device)
        
        # 입력 데이터 정보 로깅
        logging.info(f"  📥 Input logits shape: {logits.shape}")
        logging.info(f"  📥 Accelerometer features shape: {acce_features.shape}")
        logging.info(f"  🎯 Using Accelerometer modality only (index 2 out of {len(individual_features)} modalities)")
        
        # Step 1: Apply L2 normalization to Accelerometer features for consistency
        acce_features_normalized = F.normalize(acce_features, p=2, dim=1)
        
        # Accelerometer features 통계 로깅
        acce_magnitude = torch.mean(torch.abs(acce_features)).item()
        acce_norm = torch.mean(torch.norm(acce_features_normalized, p=2, dim=1)).item()
        
        logging.info(f"  ✨ Accelerometer feature magnitude: {acce_magnitude:.4f}")
        logging.info(f"  ✨ Accelerometer feature L2 norm: {acce_norm:.4f}")
        
        # Step 2: Compute LTS scale from Accelerometer features
        lts_scale = self.lts_scale(acce_features_normalized)
        
        # 🔍 DEBUG: LTS_Acce_Only scale statistics
        logging.info(f"  🔍 [DEBUG] LTS_Acce_Only scale - Mean: {lts_scale.mean().item():.6f}, Std: {lts_scale.std().item():.6f}")
        logging.info(f"  🔍 [DEBUG] Accelerometer features - Mean: {acce_features_normalized.mean().item():.6f}, Std: {acce_features_normalized.std().item():.6f}")
        
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
        
        logging.info(f"  🎯 Final OOD scores (Acce Only) - Avg: {avg_score:.4f}, Min: {min_score:.4f}, Max: {max_score:.4f}")
        logging.info("✅ [LTS_Acce_Only] Computation completed")
        
        return final_scores
