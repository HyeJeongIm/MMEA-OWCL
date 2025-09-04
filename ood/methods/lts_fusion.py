import torch
import torch.nn.functional as F
import numpy as np
import logging
from .base_ood import BaseOODDetector
from .energy import EnergyDetector

class LTSFusionDetector(BaseOODDetector):
    """LTS (Large-scale Temperature Scaling) using Fusion Features"""
    
    def __init__(self, model, device='cuda', temperature=1.0, percentile=65):
        super().__init__(model, device)
        self.temperature = temperature
        self.percentile = percentile
        
        # Energy detector를 내부적으로 사용
        self.energy_detector = EnergyDetector(model, device, temperature)
        
    def lts_scale(self, x, percentile=None):
        """
        Compute LTS scale for fusion features
        Args:
            x: fusion features tensor (batch_size, feature_dim)
            percentile: percentile for feature selection
        Returns:
            scale: LTS scale tensor (batch_size, 1)
        """
        if percentile is None:
            percentile = self.percentile
            
        assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
        assert 0 <= percentile <= 100, f"Percentile must be in [0, 100], got {percentile}"
        
        logging.info("  ⚡ Computing LTS scale from fusion features:")
        
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
        min_scale = scale.min().item()
        max_scale = scale.max().item()
        
        logging.info(f"    📈 Percentile: {percentile}%, Selected features: {k}/{n}")
        logging.info(f"    📊 S1 (all features sum): {avg_s1:.4f}")
        logging.info(f"    📊 S2 (top-k features sum): {avg_s2:.4f}")
        logging.info(f"    🎯 LTS Scale - Avg: {avg_scale:.4f}, Min: {min_scale:.4f}, Max: {max_scale:.4f}")
        
        return scale
    
    def _compute_scores_from_logits(self, logits):
        """
        Fallback: use regular energy score when fusion features not available
        """
        return self.energy_detector._compute_scores_from_logits(logits)
    
    def compute_scores_with_fusion_features(self, logits, fusion_features):
        """
        Main method: Compute LTS scores using fusion features
        Args:
            logits: (batch_size, num_classes) - pre-computed logits
            fusion_features: (segments*batch_size, feature_dim) or (batch_size, feature_dim) - fusion features from network
        Returns:
            scores: numpy array of shape (batch_size,)
        """
        logging.info("🚀 [LTS_Fusion] Starting computation with fusion features")
        
        # Ensure all tensors are on the same device
        device = logits.device
        fusion_features = fusion_features.to(device)
        
        # 입력 데이터 정보 로깅
        logging.info(f"  📥 Input logits shape: {logits.shape}")
        logging.info(f"  📥 Fusion features shape: {fusion_features.shape}")
        
        # Handle segment-based features (TBN case)
        batch_size = logits.shape[0]
        if fusion_features.shape[0] != batch_size:
            # Features are segment-based, need to aggregate to video-level
            num_segments = fusion_features.shape[0] // batch_size
            logging.info(f"  🔄 Detected segment-based features: {num_segments} segments per video")
            logging.info(f"  🔄 Reshaping from ({fusion_features.shape[0]}, {fusion_features.shape[1]}) to ({batch_size}, {num_segments}, {fusion_features.shape[1]})")
            
            # Reshape and aggregate segments
            fusion_features = fusion_features.view(batch_size, num_segments, -1)  # (batch_size, num_segments, feature_dim)
            fusion_features = fusion_features.mean(dim=1)  # (batch_size, feature_dim) - average across segments
            
            logging.info(f"  ✅ Aggregated fusion features shape: {fusion_features.shape}")
        
        # Step 1: Compute LTS scale from fusion features
        lts_scale = self.lts_scale(fusion_features)
        
        # Step 2: Apply LTS scale to logits
        scaled_logits = logits * lts_scale
        
        # Scaled logits 통계 로깅
        original_logits_mean = logits.mean().item()
        scaled_logits_mean = scaled_logits.mean().item()
        scaling_factor = scaled_logits_mean / (original_logits_mean + 1e-8)
        
        logging.info(f"  📊 Original logits mean: {original_logits_mean:.4f}")
        logging.info(f"  📊 Scaled logits mean: {scaled_logits_mean:.4f}")
        logging.info(f"  📊 Effective scaling factor: {scaling_factor:.4f}")
        
        # Step 3: Use Energy-based method to compute final OOD scores
        final_scores = self.energy_detector._compute_scores_from_logits(scaled_logits)
        
        # 최종 점수 통계 로깅
        avg_score = np.mean(final_scores)
        min_score = np.min(final_scores)
        max_score = np.max(final_scores)
        
        logging.info(f"  🎯 Final OOD scores - Avg: {avg_score:.4f}, Min: {min_score:.4f}, Max: {max_score:.4f}")
        logging.info("✅ [LTS_Fusion] Computation completed")
        
        return final_scores

    def compute_scores_from_cached_data(self, logits, fusion_features):
        """
        Compute scores using cached logits and fusion features
        This method is called by the evaluation framework
        """
        return self.compute_scores_with_fusion_features(logits, fusion_features)
