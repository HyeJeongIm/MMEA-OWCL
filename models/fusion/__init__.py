from .fusion_concat import FusionConcat
from .fusion_cmr import FusionCMR
from .mand_fusion import MANDFusion

def get_fusion(midfusion, feature_dim, modality, dropout, num_segments=None, shared_dim=256, num_classes=None,
               consensus_type='avg', before_softmax=True, pretrain_epochs=None, aux_loss_weight=0.5, **kwargs):
    if midfusion == "concat":
        return FusionConcat(feature_dim, modality, dropout)

    elif midfusion == "attention":
        return FusionCMR(
            input_dim=feature_dim,
            modality=modality,
            fusion_type="attention",
            dropout=dropout,
            num_segments=num_segments
        )

    elif midfusion == "mand_fusion":
        return MANDFusion(
            feature_dim=feature_dim,
            modality=modality,
            dropout=dropout,
            num_classes=num_classes or 100,
            consensus_type=consensus_type,
            before_softmax=before_softmax,
            num_segments=num_segments or 8,
            pretrain_epochs=pretrain_epochs if pretrain_epochs is not None else 5,
            aux_loss_weight=aux_loss_weight,
        )

    else:
        raise ValueError(f"Unknown midfusion type: {midfusion}")
