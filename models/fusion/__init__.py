from .fusion_concat import FusionConcat
from .fusion_cmr import FusionCMR
# from .fusion_context_gating import FusionContextGating
# from .fusion_multimodal_gating import FusionMultimodalGating

def get_fusion(midfusion, feature_dim, modality, dropout, num_segments=None):
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

    else:
        raise ValueError(f"Unknown midfusion type: {midfusion}")
