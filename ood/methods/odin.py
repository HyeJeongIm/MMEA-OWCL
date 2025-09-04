import torch
import torch.nn.functional as F
from .base_ood import BaseOODDetector

class ODINDetector(BaseOODDetector):
    """ODIN: Out-of-Distribution Detector (Simplified without gradient perturbation)"""
    
    def __init__(self, model, device='cuda', temperature=1000.0):
        super().__init__(model, device)
        self.temperature = temperature
        # 🗑️ REMOVED: epsilon - 현재 구현에서 사용되지 않음
    
    def _compute_scores_from_logits(self, logits):
        """Compute ODIN scores: max softmax probability with temperature scaling"""
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=1)
        return probs.max(1)[0].cpu().numpy()