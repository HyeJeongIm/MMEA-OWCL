import torch
import torch.nn.functional as F
from .base_ood import BaseOODDetector

class MSPDetector(BaseOODDetector):
    """Maximum Softmax Probability (MSP) Detector"""
    
    def _compute_scores_from_logits(self, logits):
        """Compute MSP scores: max softmax probability"""
        probs = F.softmax(logits, dim=1)
        return probs.max(1)[0].cpu().numpy()