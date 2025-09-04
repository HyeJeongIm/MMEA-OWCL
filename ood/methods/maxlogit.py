import torch
from .base_ood import BaseOODDetector

class MaxLogitDetector(BaseOODDetector):
    """MaxLogit-based Out-of-Distribution Detector"""
    
    def _compute_scores_from_logits(self, logits):
        """Compute max logit scores"""
        max_logits = torch.max(logits, dim=1)[0]
        return max_logits.cpu().numpy()
