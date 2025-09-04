import torch
import torch.nn.functional as F
from .base_ood import BaseOODDetector

class KLDivergenceDetector(BaseOODDetector):
    """KL Divergence-based Out-of-Distribution Detector"""
    
    def __init__(self, model, device='cuda', temperature=1.0, uniform_prior=True):
        super().__init__(model, device)
        self.temperature = temperature
        self.uniform_prior = uniform_prior
    
    def _compute_scores_from_logits(self, logits):
        """Compute KL divergence scores from uniform prior"""
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=1)
        
        if self.uniform_prior:
            # KL divergence from uniform distribution
            num_classes = probs.shape[1]
            uniform_probs = torch.ones_like(probs) / num_classes
            kl_div = F.kl_div(probs.log(), uniform_probs, reduction='none').sum(dim=1)
        else:
            # KL divergence from predicted distribution
            kl_div = torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        return kl_div.cpu().numpy()
