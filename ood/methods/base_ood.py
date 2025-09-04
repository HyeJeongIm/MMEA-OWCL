import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod

class BaseOODDetector(ABC):
    """Base class for OOD detection methods"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.name = self.__class__.__name__
    
    def compute_scores_from_cached_logits(self, logits):
        """Compute OOD scores from pre-extracted logits (optimized path)"""
        return self._compute_scores_from_logits(logits)

    @abstractmethod
    def _compute_scores_from_logits(self, logits):
        """Compute scores from logits - to be implemented by subclasses"""
        pass