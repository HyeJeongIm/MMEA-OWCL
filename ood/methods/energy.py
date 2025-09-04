import torch
from .base_ood import BaseOODDetector

class EnergyDetector(BaseOODDetector):
    """Energy-based Out-of-Distribution Detection"""
    
    def __init__(self, model, device='cuda', temperature=1.0):
        super().__init__(model, device)
        self.temperature = temperature
    
    def _compute_scores_from_logits(self, logits):
        """Compute Energy scores: T * logsumexp(logits/T)"""
        energy = self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        return energy.cpu().numpy()