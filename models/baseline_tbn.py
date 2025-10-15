# models/baseline.py

import torch
import torch.nn as nn
import copy
import logging

from models.backbones import get_backbone
from models.fusion import get_fusion
from models.classifier.classification_tbn import TBNClassification


class TBNBaseline(nn.Module):
    """Multi-modal baseline network with backbone, fusion, and classifier"""
    
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_segments = args["num_segments"]
        self.modality = args["modality"]
        self.backbone_name = args["backbone"]  # e.g., 'tbn'
        self.fusion_type = args.get("fusion_type", args.get("midfusion", "concat"))  # 'concat', 'attention', etc.
        self.dropout = args["dropout"]
        self.consensus_type = args["consensus_type"]
        self.before_softmax = args["before_softmax"]

        if not self.before_softmax and self.consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        # Initialize backbone network for feature extraction
        self.backbone = get_backbone(args)  # output: feature list per modality

        # Initialize fusion network to combine multi-modal features
        self.fusion = get_fusion(
            midfusion=self.fusion_type,
            feature_dim=self.backbone.feature_dim, # 각 모달리티마다 1024
            modality=self.modality,
            dropout=self.dropout,
            num_segments=self.num_segments,
            shared_dim=args.get("shared_dim", 256),  # JSON에서 설정 가능
            num_classes=args.get("init_cls", 8),  # 초기 클래스 수
            consensus_type=self.consensus_type,  # TBN consensus 방법
            before_softmax=self.before_softmax,   # TBN softmax 옵션
            pretrain_epochs=args.get("pretrain_epochs", None),  # Auxiliary head pretrain epochs (JSON에서 설정 가능)
            confidence_method=args.get("confidence_method", "max_prob")  # Confidence 계산 방법 (JSON에서 설정 가능)
        )

        # Set final feature dimension based on modality count
        # Follows the same logic as original Baseline class
        if len(self.modality) > 1:
            self.feature_dim = 512  # Multi-modal fusion output
        else:
            self.feature_dim = self.backbone.feature_dim  # Single modality: keep original dimension
            
        # Debug: Print feature dimension info
        print(f"🔍 BaselineTBN Debug:")
        print(f"   Modality count: {len(self.modality)}")
        print(f"   Backbone feature_dim: {self.backbone.feature_dim}")
        print(f"   After fusion feature_dim: {self.feature_dim}")
        self.fc = None  # Classifier will be created via update_fc()

        # Pass num_segments to fusion if it supports TBN (auxiliary_head, auxiliary_head_v2 등)
        if hasattr(self.fusion, 'num_segments'):
            self.fusion.num_segments = self.num_segments
            print(f"🔧 Set fusion.num_segments = {self.num_segments}")

        print("=" * 40)
        print("✅ Baseline Model Configuration")
        print("-" * 40)
        print(f"  Backbone:        {self.backbone_name}")
        print(f"  Fusion:          {self.fusion_type}")
        print(f"  Modality:        {self.modality}")
        print(f"  Segments:        {self.num_segments}")
        print(f"  Dropout:         {self.dropout}")
        print(f"  Consensus:       {self.consensus_type}")
        print("=" * 40)

    @property
    def output_dim(self):
        """Return output feature dimension"""
        return self.feature_dim

    def extract_vector(self, x):
        """Forward pass: backbone -> fusion"""
        features = self.backbone(x)  # Extract per-modality features
        fused = self.fusion(features)  # Fuse multi-modal features
        return fused["features"]

    def forward(self, x, targets=None):
        """Forward pass: backbone -> fusion -> classifier"""
        features = self.backbone(x)  # Extract features from each modality
        fused = self.fusion(features, targets=targets)  # Combine features across modalities (with targets for auxiliary loss)
        out = self.fc(fused["features"])  # Apply classifier
        out.update(fused)  # Include fusion output
        
        # 🎯 Auxiliary loss를 최상위로 이동 (학습 루프에서 쉽게 접근 가능)
        if 'auxiliary_loss' in fused:
            out['auxiliary_loss'] = fused['auxiliary_loss']
            out['aux_loss_weight'] = fused.get('aux_loss_weight', 0.0)
        
        return out

    def update_fc(self, nb_classes):
        """Update classifier for new number of classes while preserving weights"""
        # Create new classifier with updated class count
        new_fc = TBNClassification(
            feature_dim=self.feature_dim,
            modality=self.modality,
            num_class=nb_classes,
            consensus_type=self.consensus_type,
            before_softmax=self.before_softmax,
            num_segments=self.num_segments
        )

        # Preserve existing classifier weights if available
        if self.fc is not None:
            nb_output = self.fc.num_class
            new_fc.fc_action.weight.data[:nb_output] = self.fc.fc_action.weight.data
            new_fc.fc_action.bias.data[:nb_output] = self.fc.fc_action.bias.data

        self.fc = new_fc
        
        # Update fusion auxiliary heads if available
        if hasattr(self.fusion, 'update_auxiliary_heads'):
            self.fusion.update_auxiliary_heads(nb_classes)
            logging.info(f"🎯 Updated fusion auxiliary heads to {nb_classes} classes")

    def copy(self):
        """Create deep copy of the model"""
        return copy.deepcopy(self)

    def freeze(self):
        """Freeze all parameters for inference"""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self
