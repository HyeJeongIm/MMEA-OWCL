import torch
import torch.nn as nn
from torch.nn.init import normal_, constant_

class FusionConcat(nn.Module):
    def __init__(self, feature_dim, modality, dropout):
        super().__init__()
        self.modality = modality
        self.dropout = dropout
        
        if len(self.modality) > 1:  # Multi-modal fusion
            input_dim = len(self.modality) * feature_dim
            self.fc1 = nn.Linear(input_dim, 512)
            self.relu = nn.ReLU()
            
            # weight init
            normal_(self.fc1.weight, 0, 0.001)
            constant_(self.fc1.bias, 0)
        
        # Dropout layer (for both multi-modal and single modality)
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, inputs, targets=None):
        """
        Args:
            inputs: List of modality features
            targets: Ground truth labels (not used in concat fusion, for interface compatibility)
        """
        if len(self.modality) > 1:  # Multi-modal fusion
            x = torch.cat(inputs, dim=1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout_layer(x)
        else:  # Single modality - pass through without fusion
            x = inputs[0]  # Keep original dimensions
            x = self.dropout_layer(x)  # Apply dropout to single modality too
            
        # Debug: Print feature dimensions
        if hasattr(self, '_debug_printed') and not self._debug_printed:
            print(f"🔍 FusionConcat Debug:")
            print(f"   Modality count: {len(self.modality)}")
            print(f"   Input shapes: {[inp.shape for inp in inputs]}")
            print(f"   Output shape: {x.shape}")
            self._debug_printed = True
            
        return {'features': x}
    
    
