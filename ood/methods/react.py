import torch
from .base_ood import BaseOODDetector

class ReActDetector(BaseOODDetector):
    """ReAct: Out-of-Distribution Detection with Rectified Activations"""
    
    def __init__(self, model, device='cuda', threshold=1.0):
        super().__init__(model, device)
        self.threshold = threshold
    
    def react(self, x, threshold=None):
        """Apply ReAct transformation: clip features at threshold"""
        if threshold is None:
            threshold = self.threshold
        x = x.clip(max=threshold)
        return x
    
    def _compute_scores_from_logits(self, logits):
        """Compute Energy scores from logits (fallback)"""
        energy = torch.logsumexp(logits, dim=1)
        return energy.cpu().numpy()

