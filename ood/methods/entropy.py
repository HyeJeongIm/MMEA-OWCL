import torch
import torch.nn.functional as F
from .base_ood import BaseOODDetector

class EntropyDetector(BaseOODDetector):
    """
    Entropy-based Out-of-Distribution Detector.
    Returns an 'ID confidence' score where higher is better for ID.
    """
    
    def __init__(self, model, device='cuda', temperature=1.0):
        super().__init__(model, device)
        self.temperature = temperature
    
    def _compute_scores_from_logits(self, logits):
        """
        Computes ID confidence scores based on normalized entropy.
        Score = 1 - Normalized Entropy
        """
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=1)
        
        # 엔트로피 계산 (불확실성)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        # 엔트로피 정규화 (logits와 같은 device 사용)
        num_classes = torch.tensor(logits.shape[1], dtype=torch.float, device=logits.device)
        normalized_entropy = entropy / torch.log(num_classes)
        
        # (수정된 부분) 불확실성을 신뢰도 점수로 변환
        confidence_scores = 1.0 - normalized_entropy
        
        return confidence_scores.cpu().numpy()