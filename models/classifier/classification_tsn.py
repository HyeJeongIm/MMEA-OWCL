# models/classifier/classification_cmr.py
# B버전 (CMR_MFN, TSN) 전용 Classification Network

import torch
from torch import nn
from torch.nn.init import normal_, constant_

class TSNClassification(nn.Module):
    """
    Classification Network for TSN
    - Uses gen_train_fc method
    - Simple structure (dropout + linear)
    - No ConsensusModule
    """
    def __init__(self, feature_dim, modality, num_class, dropout, num_segments):
        super().__init__()
        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.dropout = dropout

        self._add_classification_layer(feature_dim)

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)

    def _add_classification_layer(self, input_dim):
        """Add linear classification layer with weight initialization"""
        std = 0.001

        self.fc_action = nn.Linear(input_dim, self.num_class)
        normal_(self.fc_action.weight, 0, std)
        constant_(self.fc_action.bias, 0)
        
        # 편의를 위해 weight와 bias에 직접 접근 가능하도록 설정
        self.weight = self.fc_action.weight
        self.bias = self.fc_action.bias

    def forward(self, inputs):
        """Forward pass: dropout -> linear -> logits"""
        if self.dropout > 0:
            base_out = self.dropout_layer(inputs)
        else:
            base_out = inputs
            
        base_out = self.fc_action(base_out)
        output = {'logits': base_out}
        return output

    # def copy_weights_from(self, other_fc, num_classes_to_copy):
    #     """
    #     기존 classifier의 weight를 복사 (gen_train_fc에서 사용)
    #     """
    #     if hasattr(other_fc, 'fc_action'):
    #         self.fc_action.weight.data[:num_classes_to_copy] = other_fc.fc_action.weight.data[:num_classes_to_copy]
    #         self.fc_action.bias.data[:num_classes_to_copy] = other_fc.fc_action.bias.data[:num_classes_to_copy]
    #     elif hasattr(other_fc, 'weight'):
    #         self.fc_action.weight.data[:num_classes_to_copy] = other_fc.weight.data[:num_classes_to_copy]
    #         self.fc_action.bias.data[:num_classes_to_copy] = other_fc.bias.data[:num_classes_to_copy]