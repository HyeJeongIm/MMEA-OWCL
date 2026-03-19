from .fusion_concat import FusionConcat
from .fusion_cmr import FusionCMR
from .imu_cosine_gate import IMUCosineGateFusion
from .imu_euclidean_gate import IMUEuclideanGateFusion
from .imu_kl_gate import IMUKLGateFusion
from .imu_entropy_gate import IMUEntropyGateFusion
from .hierarchical_concat_fusion import HierarchicalConcatFusion
from .auxiliary_head_fusion import AuxiliaryHeadFusion
from .auxiliary_head_fusion_v2 import AuxiliaryHeadFusionV2
from .auxiliary_head_fusion_v2_3 import AuxiliaryHeadFusionV2_3
from .auxiliary_head_fusion_v2_4 import AuxiliaryHeadFusionV2_4
from .auxiliary_head_fusion_v2_5 import AuxiliaryHeadFusionV2_5
from .auxiliary_head_fusion_v2_6 import AuxiliaryHeadFusionV2_6
from .mand_fusion import MANDFusion
from .auxiliary_head_fusion_v2_8 import AuxiliaryHeadFusionV2_8
from .auxiliary_head_fusion_v2_9 import AuxiliaryHeadFusionV2_9
from .auxiliary_head_fusion_v2_10 import AuxiliaryHeadFusionV2_10
from .gated_cross_modal_fusion import GatedCrossModalFusion
from .cross_attention_fusion import CrossAttentionFusion
# from .fusion_context_gating import FusionContextGating
# from .fusion_multimodal_gating import FusionMultimodalGating

def get_fusion(midfusion, feature_dim, modality, dropout, num_segments=None, shared_dim=256, num_classes=None, 
               consensus_type='avg', before_softmax=True, pretrain_epochs=None, confidence_method="max_prob", aux_loss_weight=0.5,
               energy_norm_method="zscore"):
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
    
    elif midfusion == "imu_cosine_gate":
        return IMUCosineGateFusion(feature_dim, modality, dropout, shared_dim=shared_dim)
    
    elif midfusion == "imu_euclidean_gate":
        return IMUEuclideanGateFusion(feature_dim, modality, dropout, shared_dim=shared_dim)
    
    elif midfusion == "imu_kl_gate":
        return IMUKLGateFusion(feature_dim, modality, dropout, shared_dim=shared_dim)
    
    elif midfusion == "imu_entropy_gate":
        return IMUEntropyGateFusion(feature_dim, modality, dropout, shared_dim=shared_dim)

    elif midfusion == "hierarchical_concat":
        return HierarchicalConcatFusion(feature_dim, modality, dropout)
    
    elif midfusion == "auxiliary_head":
        return AuxiliaryHeadFusion(feature_dim, modality, dropout, num_classes or 100)
    
    elif midfusion == "auxiliary_head_v2":
        return AuxiliaryHeadFusionV2(
            feature_dim=feature_dim, 
            modality=modality, 
            dropout=dropout, 
            num_classes=num_classes or 100,
            consensus_type=consensus_type,
            before_softmax=before_softmax,
            num_segments=num_segments or 8,
            aux_loss_weight=aux_loss_weight,
        )
    
    elif midfusion == "auxiliary_head_v2_3":
        return AuxiliaryHeadFusionV2_3(
            feature_dim=feature_dim, 
            modality=modality, 
            dropout=dropout, 
            num_classes=num_classes or 100,
            consensus_type=consensus_type,
            before_softmax=before_softmax,
            num_segments=num_segments or 8,
            aux_loss_weight=aux_loss_weight,
        )
    
    elif midfusion == "auxiliary_head_v2_4":
        return AuxiliaryHeadFusionV2_4(
            feature_dim=feature_dim, 
            modality=modality, 
            dropout=dropout, 
            num_classes=num_classes or 100,
            consensus_type=consensus_type,
            before_softmax=before_softmax,
            num_segments=num_segments or 8,
            warmup_epochs=pretrain_epochs if pretrain_epochs is not None else 5,  # JSON에서 설정 가능
            aux_loss_weight=aux_loss_weight,
        )
        
    elif midfusion == "auxiliary_head_v2_5":
        return AuxiliaryHeadFusionV2_5(
            feature_dim=feature_dim, 
            modality=modality, 
            dropout=dropout, 
            num_classes=num_classes or 100,
            consensus_type=consensus_type,
            before_softmax=before_softmax,
            num_segments=num_segments or 8,
            warmup_epochs=pretrain_epochs if pretrain_epochs is not None else 1,  # JSON에서 설정 가능 (기본값 1)
            aux_loss_weight=aux_loss_weight,
        )
    
    elif midfusion == "auxiliary_head_v2_6":
        return AuxiliaryHeadFusionV2_6(
            feature_dim=feature_dim, 
            modality=modality, 
            dropout=dropout, 
            num_classes=num_classes or 100,
            consensus_type=consensus_type,
            before_softmax=before_softmax,
            num_segments=num_segments or 8,
            pretrain_epochs=pretrain_epochs if pretrain_epochs is not None else 5,  # JSON에서 설정 가능 (기본값 5)
            aux_loss_weight=aux_loss_weight,
        )
    
    elif midfusion == "mand_fusion":
        return MANDFusion(
            feature_dim=feature_dim,
            modality=modality,
            dropout=dropout,
            num_classes=num_classes or 100,
            confidence_method=confidence_method,
            consensus_type=consensus_type,
            before_softmax=before_softmax,
            num_segments=num_segments or 8,
            pretrain_epochs=pretrain_epochs if pretrain_epochs is not None else 5,
            aux_loss_weight=aux_loss_weight,
            energy_norm_method=energy_norm_method,
        )
    
    elif midfusion == "auxiliary_head_v2_8":
        return AuxiliaryHeadFusionV2_8(
            feature_dim=feature_dim, 
            modality=modality, 
            dropout=dropout, 
            num_classes=num_classes or 100,
            confidence_method=confidence_method,  # JSON에서 설정 가능 (기본값: max_prob)
            consensus_type=consensus_type,
            before_softmax=before_softmax,
            num_segments=num_segments or 8,
            pretrain_epochs=pretrain_epochs if pretrain_epochs is not None else 5,  # JSON에서 설정 가능 (기본값 5)
            aux_loss_weight=aux_loss_weight,
        )
    
    elif midfusion == "auxiliary_head_v2_9":
        return AuxiliaryHeadFusionV2_9(
            feature_dim=feature_dim, 
            modality=modality, 
            dropout=dropout, 
            num_classes=num_classes or 100,
            confidence_method=confidence_method,  # JSON에서 설정 가능 (기본값: max_prob)
            consensus_type=consensus_type,
            before_softmax=before_softmax,
            num_segments=num_segments or 8,
            pretrain_epochs=pretrain_epochs if pretrain_epochs is not None else 5,  # JSON에서 설정 가능 (기본값 5)
            aux_loss_weight=aux_loss_weight,
        )
    
    elif midfusion == "auxiliary_head_v2_10":
        return AuxiliaryHeadFusionV2_10(
            feature_dim=feature_dim, 
            modality=modality, 
            dropout=dropout, 
            num_classes=num_classes or 100,
            confidence_method=confidence_method,  # JSON에서 설정 가능 (기본값: max_prob)
            consensus_type=consensus_type,
            before_softmax=before_softmax,
            num_segments=num_segments or 8,
            pretrain_epochs=pretrain_epochs if pretrain_epochs is not None else 5,  # JSON에서 설정 가능 (기본값 5)
            aux_loss_weight=aux_loss_weight,
        )
    
    elif midfusion == "gated_cross_modal":
        return GatedCrossModalFusion(feature_dim, modality, dropout)
    
    elif midfusion == "cross_attention":
        return CrossAttentionFusion(feature_dim, modality, dropout)

    else:
        raise ValueError(f"Unknown midfusion type: {midfusion}")
