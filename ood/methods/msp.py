import torch
import torch.nn.functional as F
from .base_ood import BaseOODDetector

class MSPDetector(BaseOODDetector):
    """
    🎯 DEPRECATED: Use MSPAuxLogitsConfRawLogitLevelDetector instead
    
    This class is kept for backward compatibility.
    The new naming convention is:
    - MSPAuxLogitsConfRawLogitLevelDetector: Raw confidence at logit level
    - MSPAuxLogitsConfNormalizedLogitLevelDetector: Normalized confidence at logit level
    - MSPAuxLogitsConfRawScoreLevelDetector: Raw confidence at score level
    - MSPAuxLogitsConfNormalizedScoreLevelDetector: Normalized confidence at score level
    """
    
    def _compute_scores_from_logits(self, logits):
        """Compute MSP scores: max softmax probability"""
        probs = F.softmax(logits, dim=1)
        return probs.max(1)[0].cpu().numpy()