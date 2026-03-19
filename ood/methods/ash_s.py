import torch
import numpy as np
from .base_ood import BaseOODDetector

class ASHSDetector(BaseOODDetector):
    """ASH-S: Adaptive Sample-wise Histogram Scaling"""
    
    def __init__(self, model, device='cuda', percentile=90):
        super().__init__(model, device)
        self.percentile = percentile
    
    def ash_s(self, x, percentile=None):
        """Apply ASH-S transformation"""
        if percentile is None:
            percentile = self.percentile
        assert x.dim() == 2
        assert 0 <= percentile <= 100
        # calculate the sum of the input per sample
        s1 = x.sum(dim=1)
        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        v, i = torch.topk(x, k, dim=1)
        x.zero_().scatter_(dim=1, index=i, src=v)
        s2 = v.sum(dim=1)
        scale = s1 / s2
        return x * torch.exp(scale[:, None])
    
    def _compute_scores_from_logits(self, logits):
        """Compute Energy scores from logits (fallback)"""
        energy = torch.logsumexp(logits, dim=1)
        return energy.cpu().numpy()

