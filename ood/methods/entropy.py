import torch
import torch.nn.functional as F
from .base_ood import BaseOODDetector

class EntropyDetector(BaseOODDetector):
    """Entropy-based Out-of-Distribution Detector"""
    
    def __init__(self, model, device='cuda', temperature=1.0):
        super().__init__(model, device)
        self.temperature = temperature
    
    def _compute_scores_from_logits(self, logits):
        """Compute entropy scores: -sum(p * log(p))"""
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy.cpu().numpy()
