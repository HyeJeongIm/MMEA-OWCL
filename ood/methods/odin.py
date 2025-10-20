import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_ood import BaseOODDetector

class ODINDetector(BaseOODDetector):
    """
    ODIN: Out-of-Distribution Detector for Neural Networks with Input perturbation
    
    Reference:
        Liang et al. "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks"
        ICLR 2018
    
    핵심 아이디어:
    1. Temperature scaling: logits / T로 scaling하여 softmax 부드럽게
    2. Input perturbation: gradient 방향으로 작은 noise 추가하여 ID confidence 증폭
    """
    
    def __init__(self, model, device='cuda', temperature=1000.0, magnitude=0.0014):
        """
        Args:
            model: 학습된 모델
            device: 디바이스
            temperature: Temperature scaling 파라미터 (default: 1000.0)
            magnitude: Input perturbation 크기 (default: 0.0014)
        """
        super().__init__(model, device)
        self.temperature = temperature
        self.magnitude = magnitude
        self.criterion = nn.CrossEntropyLoss()
    
    def odin_score(self, inputs, temperature=None, magnitude=None):
        """
        Apply ODIN method: temperature scaling + input perturbation
        
        Args:
            inputs: Input tensor [batch_size, ...]
            temperature: Temperature for scaling (optional)
            magnitude: Perturbation magnitude (optional)
        
        Returns:
            scores: ODIN scores (max softmax probability) [batch_size]
        """
        if temperature is None:
            temperature = self.temperature
        if magnitude is None:
            magnitude = self.magnitude
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # 1. Forward pass with gradient
        inputs_var = torch.autograd.Variable(inputs, requires_grad=True)
        outputs = self.model(inputs_var)
        
        # Handle dict output (from multi-modal models)
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        # Get predicted labels
        max_indices = torch.argmax(logits, dim=1)
        
        # 2. Temperature scaling
        scaled_logits = logits / temperature
        
        # 3. Calculate loss for gradient
        labels = torch.autograd.Variable(max_indices)
        loss = self.criterion(scaled_logits, labels)
        loss.backward()
        
        # 4. Compute gradient direction
        # Normalizing the gradient to binary in {0, 1} → {-1, 1}
        gradient = torch.ge(inputs_var.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        
        # 5. Add perturbation to inputs
        perturbed_inputs = inputs_var.data + (-magnitude * gradient)
        
        # 6. Forward pass with perturbed inputs (no grad)
        with torch.no_grad():
            perturbed_outputs = self.model(perturbed_inputs)
            
            # Handle dict output
            if isinstance(perturbed_outputs, dict):
                perturbed_logits = perturbed_outputs['logits']
            else:
                perturbed_logits = perturbed_outputs
            
            # Temperature scaling
            perturbed_scaled_logits = perturbed_logits / temperature
            
            # 7. Calculate softmax probabilities
            # Manual softmax for numerical stability
            logits_np = perturbed_scaled_logits.cpu().numpy()
            logits_np = logits_np - np.max(logits_np, axis=1, keepdims=True)
            exp_logits = np.exp(logits_np)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # 8. Max probability as score
            scores = np.max(probs, axis=1)
        
        return scores
    
    def _compute_scores_from_logits(self, logits):
        """
        Fallback: Compute ODIN scores from pre-extracted logits
        (without input perturbation, only temperature scaling)
        """
        # Temperature scaling
        scaled_logits = logits / self.temperature
        
        # Softmax
        probs = F.softmax(scaled_logits, dim=1)
        
        # Max probability
        max_probs = probs.max(1)[0]
        
        return max_probs.cpu().numpy()
