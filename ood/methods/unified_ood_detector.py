import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from .base_ood import BaseOODDetector
from .msp import MSPDetector
from .energy import EnergyDetector
from .maxlogit import MaxLogitDetector
from .lts_fusion import LTSFusionDetector
from .react import ReActDetector
from .scale import ScaleDetector
from .ash_s import ASHSDetector
from .odin import ODINDetector
from .entropy import EntropyDetector


class UnifiedOODDetector(BaseOODDetector):
    """
    🎯 Unified OOD Detector
    
    통합 OOD 검출기로, 설정(config)에 따라 다양한 방법론과 모드를 지원합니다.
    
    지원하는 방법론 (method):
        - 'msp': Maximum Softmax Probability
        - 'energy': Energy-based OOD Detection
        - 'maxlogit': Maximum Logit
        - 'lts': Large-scale Temperature Scaling (fusion features 필요)
        - 'react': ReAct - Rectified Activations (features 필요)
        - 'scale': Scale - Exponential Scaling (features 필요)
        - 'ash_s': ASH-S - Adaptive Histogram Scaling (features 필요)
        - 'odin': ODIN - Input Perturbation + Temperature (raw inputs 권장)
        - 'entropy': Entropy-based OOD Detection
    
    지원하는 모드 (mode):
        - 'baseline': Main logits만 사용
        - 'hybrid_noweight': Main logits (가중치 1) + Auxiliary logits (가중치 1씩) 균등 합산
        - 'hybrid_conf_raw': Main logits (가중치 1) + Auxiliary logits (raw confidence로 가중)
        - 'hybrid_conf_normalized': Main logits (가중치 1) + Auxiliary logits (normalized confidence로 가중)
    
    Hybrid 모드 핵심:
        - Main logits는 **항상 가중치 1**로 포함
        - Auxiliary logits는 모드에 따라 다른 가중치 적용
        - 최종 fused logits = Main (×1) + Σ(Auxiliary_i × weight_i)
    
    사용 예시:
        # 방법 1: Config로 직접 생성
        detector = UnifiedOODDetector(model, config={'method': 'msp', 'mode': 'baseline'})
        
        # 방법 2: Method name으로 자동 생성 (추천)
        detector = UnifiedOODDetector.from_method_name(model, 'MSP_Baseline')
        detector = UnifiedOODDetector.from_method_name(model, 'Energy_Hybrid_ConfRaw')
        detector = UnifiedOODDetector.from_method_name(model, 'MaxLogit_Hybrid_NoWeight')
    """
    
    VALID_METHODS = ['msp', 'energy', 'maxlogit', 'lts', 'react', 'scale', 'ash_s', 'odin', 'entropy']
    VALID_MODES = ['baseline', 'hybrid_noweight', 'hybrid_conf_raw', 'hybrid_conf_normalized']
    
    # Method name to config 매핑
    METHOD_NAME_MAPPING = {
        # Baseline으로 끝나는 경우
        'baseline': 'baseline',
        # Hybrid로 시작하는 경우
        'noweight': 'hybrid_noweight',
        'confraw': 'hybrid_conf_raw',
        'confnormalized': 'hybrid_conf_normalized',
    }
    
    def __init__(self, model, device='cuda', config=None):
        """
        Args:
            model: 모델 인스턴스
            device: 디바이스 ('cuda' or 'cpu')
            config: 설정 딕셔너리
                - method: OOD 방법론 ('msp', 'energy', 'maxlogit')
                - mode: 동작 모드 ('baseline', 'hybrid_noweight', 'hybrid_conf_raw', 'hybrid_conf_normalized')
                - temperature: Energy용 temperature 파라미터 (기본값: 1.0)
        """
        super().__init__(model, device)
        
        if config is None:
            config = {'method': 'msp', 'mode': 'baseline'}
        
        self.config = config
        self.method = config.get('method', 'msp').lower()
        self.mode = config.get('mode', 'baseline').lower()
        self.temperature = config.get('temperature', 1.0)
        
        # 유효성 검사
        if self.method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method: {self.method}. Must be one of {self.VALID_METHODS}")
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {self.VALID_MODES}")
        
        # 클래스 이름 업데이트 (디버깅용)
        self.name = f"UnifiedOODDetector_{self.method}_{self.mode}"
        
        # 기존 detector 인스턴스 생성 (코드 재사용)
        self._base_detector = self._create_base_detector()
    
    @classmethod
    def from_method_name(cls, model, method_name, device='cuda'):
        """
        Method name으로부터 UnifiedOODDetector 인스턴스 생성
        
        Method name 형식:
            - {Method}_{Mode}
            - 예: MSP_Baseline, Energy_Hybrid_ConfRaw, MaxLogit_Hybrid_NoWeight
        
        Args:
            model: 모델 인스턴스
            method_name: OOD method 이름 (예: "MSP_Baseline", "Energy_Hybrid_ConfRaw")
            device: 디바이스 ('cuda' or 'cpu')
        
        Returns:
            UnifiedOODDetector 인스턴스
        
        Examples:
            >>> detector = UnifiedOODDetector.from_method_name(model, "MSP_Baseline")
            >>> detector = UnifiedOODDetector.from_method_name(model, "Energy_Hybrid_ConfRaw")
            >>> detector = UnifiedOODDetector.from_method_name(model, "MaxLogit_Hybrid_NoWeight")
        """
        config = cls.parse_method_name(method_name)
        return cls(model, device=device, config=config)
    
    @staticmethod
    def parse_method_name(method_name):
        """
        Method name을 파싱하여 config 딕셔너리 생성
        
        Args:
            method_name: OOD method 이름 (예: "MSP_Baseline", "Energy_Hybrid_ConfRaw")
        
        Returns:
            config: {'method': str, 'mode': str, 'temperature': float (optional)}
        
        Raises:
            ValueError: 잘못된 method name 형식
        
        Examples:
            >>> UnifiedOODDetector.parse_method_name("MSP_Baseline")
            {'method': 'msp', 'mode': 'baseline'}
            
            >>> UnifiedOODDetector.parse_method_name("Energy_Hybrid_ConfRaw")
            {'method': 'energy', 'mode': 'hybrid_conf_raw', 'temperature': 1.0}
            
            >>> UnifiedOODDetector.parse_method_name("MaxLogit_Hybrid_NoWeight")
            {'method': 'maxlogit', 'mode': 'hybrid_noweight'}
        """
        # Method name을 언더스코어로 분리
        parts = method_name.split('_')
        
        if len(parts) < 2:
            raise ValueError(f"Invalid method name format: {method_name}. Expected format: {{Method}}_{{Mode}}")
        
        # 특수 케이스: ASH_S (두 부분으로 이루어진 method name)
        if len(parts) >= 3 and parts[0].upper() == 'ASH' and parts[1].upper() == 'S':
            method = 'ash_s'
            mode_parts = parts[2:]  # ['Baseline'] or ['Hybrid', ...]
        else:
            # 일반적인 경우: 첫 번째 부분이 method
            method = parts[0].lower()
            mode_parts = parts[1:]
        
        # Mode 결정
        if len(mode_parts) == 1 and mode_parts[0].lower() == 'baseline':
            mode = 'baseline'
        elif len(mode_parts) >= 2 and mode_parts[0].lower() == 'hybrid':
            # Hybrid_NoWeight, Hybrid_ConfRaw, Hybrid_ConfNormalized
            mode_suffix = ''.join(mode_parts[1:]).lower()  # "noweight", "confraw", "confnormalized"
            
            if mode_suffix == 'noweight':
                mode = 'hybrid_noweight'
            elif mode_suffix == 'confraw':
                mode = 'hybrid_conf_raw'
            elif mode_suffix == 'confnormalized':
                mode = 'hybrid_conf_normalized'
            else:
                raise ValueError(f"Unknown hybrid mode: {mode_suffix}")
        else:
            raise ValueError(f"Invalid mode format in method name: {method_name}")
        
        # Config 생성
        config = {
            'method': method,
            'mode': mode
        }
        
        # Energy의 경우 temperature 추가
        if method == 'energy':
            config['temperature'] = 1.0
        
        return config
    
    def _create_base_detector(self):
        """
        방법론에 맞는 기존 detector 인스턴스 생성
        
        Returns:
            detector: 해당 방법론의 detector 인스턴스
        """
        if self.method == 'msp':
            return MSPDetector(self.model, self.device)
        elif self.method == 'energy':
            return EnergyDetector(self.model, self.device, temperature=self.temperature)
        elif self.method == 'maxlogit':
            return MaxLogitDetector(self.model, self.device)
        elif self.method == 'lts':
            return LTSFusionDetector(self.model, self.device, temperature=self.temperature)
        elif self.method == 'react':
            threshold = self.config.get('threshold', 1.0)
            return ReActDetector(self.model, self.device, threshold=threshold)
        elif self.method == 'scale':
            percentile = self.config.get('percentile', 90)
            return ScaleDetector(self.model, self.device, percentile=percentile)
        elif self.method == 'ash_s':
            percentile = self.config.get('percentile', 90)
            return ASHSDetector(self.model, self.device, percentile=percentile)
        elif self.method == 'odin':
            temperature = self.config.get('temperature', 1000.0)
            magnitude = self.config.get('magnitude', 0.0014)
            return ODINDetector(self.model, self.device, temperature=temperature, magnitude=magnitude)
        elif self.method == 'entropy':
            return EntropyDetector(self.model, self.device, temperature=self.temperature)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def compute_scores_from_outputs(self, outputs):
        """
        Outputs에서 OOD scores 계산
        
        Args:
            outputs: 모델의 forward 출력 딕셔너리
                - logits: Main logits [batch, num_classes] (필수)
                - auxiliary_logits: {modality: tensor [batch, num_classes]} (hybrid modes에서 필수)
                - confidences: {modality: tensor [batch]} (hybrid confidence modes에서 필수)
                - fusion_features: tensor [batch, feature_dim] (LTS, feature transforms에서 필수)
                - individual_features: dict/list of tensors (LTS hybrid modes에서 필수)
                - raw_inputs: tensor (ODIN에서 필요, optional)
        
        Returns:
            scores: OOD scores (numpy array)
        
        Raises:
            ValueError: 필요한 입력이 없을 때
        """
        # LTS 메서드 처리
        if self.method == 'lts':
            return self._compute_lts_scores(outputs)
        
        # Feature transform methods (ReAct, Scale, ASH_S)
        if self.method in ['react', 'scale', 'ash_s']:
            return self._compute_feature_transform_scores(outputs)
        
        # ODIN method (special handling)
        if self.method == 'odin':
            return self._compute_odin_scores(outputs)
        
        # Logit-level methods (MSP, Energy, MaxLogit, Entropy)
        if self.mode == 'baseline':
            # Baseline: Main logits만 사용
            if 'logits' not in outputs or outputs['logits'] is None:
                raise ValueError(f"Baseline mode requires 'logits' in outputs, but got: {outputs.keys()}")
            return self._compute_scores_from_logits(outputs['logits'])
        
        else:
            # Hybrid modes: Main + Auxiliary logits 사용
            return self._compute_hybrid_scores(outputs)
    
    def _compute_hybrid_scores(self, outputs):
        """
        Hybrid 모드에서 OOD scores 계산
        
        Hybrid = Main logits (가중치 1) + Auxiliary logits (가중치 적용)
        
        Args:
            outputs: 모델의 forward 출력 딕셔너리
        
        Returns:
            scores: OOD scores (numpy array)

        
        Raises:
            ValueError: 필요한 입력이 없을 때
        """
        main_logits = outputs.get('logits', None)
        auxiliary_logits = outputs.get('auxiliary_logits', {})
        confidences = outputs.get('confidences', {})
        
        # Hybrid 모드는 main logits 필수
        if main_logits is None:
            raise ValueError(f"Hybrid mode requires 'logits' (main logits) in outputs, but got: {outputs.keys()}")
        
        # Hybrid 모드는 auxiliary logits 필수
        if not auxiliary_logits:
            raise ValueError(f"Hybrid mode requires 'auxiliary_logits' in outputs, but got: {outputs.keys()}")
        
        # 모달리티 순서 일관성 유지
        modalities = ['RGB', 'Gyro', 'Acce']
        
        # Logit level fusion: main + auxiliary
        if self.mode == 'hybrid_noweight':
            # No weight: main (가중치 1) + auxiliary (가중치 1씩)
            fused_logits = self._fuse_logits_noweight(main_logits, auxiliary_logits, modalities)
        
        elif self.mode == 'hybrid_conf_raw':
            # Raw confidence: main (가중치 1) + auxiliary (raw confidence로 가중)
            if not confidences:
                raise ValueError(f"Hybrid_ConfRaw mode requires 'confidences' in outputs, but got: {outputs.keys()}")
            fused_logits = self._fuse_logits_conf_raw(main_logits, auxiliary_logits, confidences, modalities)
        
        elif self.mode == 'hybrid_conf_normalized':
            # Normalized confidence: main (가중치 1) + auxiliary (normalized confidence로 가중)
            if not confidences:
                raise ValueError(f"Hybrid_ConfNormalized mode requires 'confidences' in outputs, but got: {outputs.keys()}")
            fused_logits = self._fuse_logits_conf_normalized(main_logits, auxiliary_logits, confidences, modalities)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Fused logits에서 OOD score 계산
        return self._compute_scores_from_logits(fused_logits)
    
    def _fuse_logits_noweight(self, main_logits, auxiliary_logits, modalities):
        """
        Main logits + Auxiliary logits를 균등 가중치로 fusion
        
        Args:
            main_logits: tensor [batch, num_classes] - 가중치 1
            auxiliary_logits: {modality: tensor [batch, num_classes]} - 각각 가중치 1
            modalities: 모달리티 리스트
        
        Returns:
            fused_logits: [batch, num_classes]
        """
        # 모든 logits 수집: main + auxiliary
        all_logits = [main_logits]
        
        for modality in modalities:
            if modality in auxiliary_logits:
                all_logits.append(auxiliary_logits[modality])
        
        # Stack: [num_sources, batch, num_classes]
        logits_stacked = torch.stack(all_logits, dim=0)
        
        # 가중합 (가중치 모두 1): [batch, num_classes]
        weights = torch.ones(logits_stacked.size(0), device=logits_stacked.device)
        fused_logits = (logits_stacked * weights.view(-1, 1, 1)).sum(dim=0)
        
        return fused_logits
    
    def _fuse_logits_conf_raw(self, main_logits, auxiliary_logits, confidences, modalities):
        """
        Main logits (가중치 1) + Auxiliary logits (raw confidence로 가중)를 fusion
        
        Args:
            main_logits: tensor [batch, num_classes] - 가중치 1
            auxiliary_logits: {modality: tensor [batch, num_classes]}
            confidences: {modality: tensor [batch]}
            modalities: 모달리티 리스트
        
        Returns:
            fused_logits: [batch, num_classes]
        """
        batch_size = main_logits.size(0)
        num_classes = main_logits.size(1)
        
        # Main logits (가중치 1)로 시작
        fused_logits = main_logits * 1.0
        
        # Auxiliary logits를 raw confidence로 가중하여 추가
        for modality in modalities:
            if modality in auxiliary_logits and modality in confidences:
                aux_logits = auxiliary_logits[modality]  # [batch, num_classes]
                conf = confidences[modality]  # [batch]
                
                # Confidence를 가중치로 적용
                weighted_logits = aux_logits * conf.unsqueeze(-1)  # [batch, num_classes]
                fused_logits = fused_logits + weighted_logits
        
        return fused_logits
    
    def _fuse_logits_conf_normalized(self, main_logits, auxiliary_logits, confidences, modalities):
        """
        Main logits (가중치 1) + Auxiliary logits (normalized confidence로 가중)를 fusion
        
        Args:
            main_logits: tensor [batch, num_classes] - 가중치 1
            auxiliary_logits: {modality: tensor [batch, num_classes]}
            confidences: {modality: tensor [batch]}
            modalities: 모달리티 리스트
        
        Returns:
            fused_logits: [batch, num_classes]
        """
        # Auxiliary confidences 수집
        conf_list = []
        aux_logits_list = []
        
        for modality in modalities:
            if modality in auxiliary_logits and modality in confidences:
                aux_logits_list.append(auxiliary_logits[modality])
                conf_list.append(confidences[modality])
        
        # Main logits (가중치 1)로 시작
        fused_logits = main_logits * 1.0
        
        # Auxiliary가 있으면 normalized confidence로 가중하여 추가
        if conf_list:
            # Stack: [num_modalities, batch]
            conf_stacked = torch.stack(conf_list, dim=0)
            # Stack: [num_modalities, batch, num_classes]
            aux_logits_stacked = torch.stack(aux_logits_list, dim=0)
            
        # Softmax로 confidence 정규화 (dim=0: 모달리티 차원에서 합=1)
        normalized_weights = torch.softmax(conf_stacked, dim=0)  # [num_modalities, batch]

        # ─── [α-Diag] modality weight 분포 진단 로깅 ────────────────────
        # 전체 실행 중 1회만 출력 (배치마다 반복 방지)
        if not getattr(self, '_alpha_logged', False):
            w_np = normalized_weights.detach().cpu().numpy()  # [M, B]
            modality_names = ['RGB', 'Gyro', 'Acce']
            logging.info("=" * 60)
            logging.info("[α-Diag] Modality Weight (α_m) Distribution")
            logging.info(f"  shape: {w_np.shape}  (modalities x batch)")
            for i, mod in enumerate(modality_names[:w_np.shape[0]]):
                w_i = w_np[i]
                logging.info(
                    f"  α_{mod}: mean={w_i.mean():.4f}, std={w_i.std():.4f}, "
                    f"min={w_i.min():.4f}, max={w_i.max():.4f}"
                )
            per_sample_std = w_np.std(axis=0)  # [B]: 샘플별 3 modality weight의 std
            logging.info(
                f"  per-sample std(α_RGB,α_Gyro,α_Acce): "
                f"mean={per_sample_std.mean():.4f}, max={per_sample_std.max():.4f}"
            )
            logging.info(
                f"  → uniform 기준: std≈0.0000  |  "
                f"완전 one-hot 기준: std≈{((2/3)**0.5)/3:.4f}"
            )
            logging.info("=" * 60)
            self._alpha_logged = True
        # ──────────────────────────────────────────────────────────────────            # 가중합 계산
            for i in range(len(aux_logits_list)):
                weighted_logits = aux_logits_stacked[i] * normalized_weights[i].unsqueeze(-1)
                fused_logits = fused_logits + weighted_logits
        
        return fused_logits
    
    # ----------------------------------------------------------------------------------
    # LTS 전용 로직 (Baseline only)
    # ----------------------------------------------------------------------------------
    def _compute_lts_scores(self, outputs):
        """
        LTS 방법론으로 OOD scores 계산 (Baseline만 지원)
        
        Args:
            outputs: 모델의 forward 출력 딕셔너리
                - logits: Main logits [batch, num_classes]
                - fusion_features: Fusion features [batch, feature_dim]
        
        Returns:
            scores: OOD scores (numpy array)
        """
        # LTS는 Baseline 모드만 지원
        if self.mode != 'baseline':
            raise ValueError(f"LTS method only supports 'baseline' mode, but got: {self.mode}")
        
        main_logits = outputs.get('logits')
        fusion_features = outputs.get('fusion_features')
        
        if main_logits is None:
            raise ValueError(f"LTS Baseline requires 'logits' in outputs, but got: {outputs.keys()}")
        if fusion_features is None:
            raise ValueError(f"LTS Baseline requires 'fusion_features' in outputs, but got: {outputs.keys()}")
        
        # LTS Fusion Detector 사용
        return self._base_detector.compute_scores_with_fusion_features(main_logits, fusion_features)
    
    def _compute_lts_score_fusion(self, outputs, alpha=1.0):
        """
        [방법 2] LTS Score Fusion 로직.
        각 Feature에서 LTS scale을 계산하고, 이를 confidence로 가중합합니다.
        
        Args:
            outputs (dict): 모델 출력. 'individual_features', 'fusion_features', 'confidences' 필요.
            alpha (float): Fused feature score에 대한 가중치.
        
        Returns:
            torch.Tensor: 최종 가중합된 LTS scale [batch_size]
        """
        individual_features = outputs.get('individual_features')
        fusion_features = outputs.get('fusion_features')
        confidences = outputs.get('confidences', {})
        
        if fusion_features is None:
            raise ValueError("LTS Score Fusion requires 'fusion_features'.")
        if individual_features is None:
            raise ValueError("LTS Score Fusion requires 'individual_features'.")
        if not confidences:
            raise ValueError("LTS Score Fusion requires 'confidences'.")

        logging.info("  🔥 LTS Score Fusion:")
        
        # 1. Fused feature로부터 LTS scale 계산
        fused_scale = self._base_detector._calculate_lts_scale(fusion_features)
        final_scale = fused_scale * alpha
        
        logging.info(f"     - Fused scale (α={alpha}): mean={fused_scale.mean():.4f}")

        # 2. 각 모달리티별 LTS scale 계산 및 가중합
        modalities = ['RGB', 'Gyro', 'Acce']
        for i, modality in enumerate(modalities):
            if modality in confidences:
                # individual_features는 dict 또는 list일 수 있음
                if isinstance(individual_features, dict):
                    features = individual_features.get(modality)
                elif isinstance(individual_features, list):
                    features = individual_features[i] if i < len(individual_features) else None
                else:
                    features = None
                
                if features is not None:
                    conf = confidences[modality]  # [batch_size]
                    
                    individual_scale = self._base_detector._calculate_lts_scale(features)  # [batch_size]
                    
                    final_scale = final_scale + (individual_scale * conf)
                    
                    logging.info(f"     - {modality} scale: mean={individual_scale.mean():.4f}, conf={conf.mean():.4f}")
        
        logging.info(f"     - Final scale: mean={final_scale.mean():.4f}")
        
        return final_scale

    def _compute_lts_feature_fusion(self, outputs, alpha=1.0):
        """
        [방법 1] LTS Feature Fusion 로직.
        각 Feature를 confidence로 가중합하여 하이브리드 Feature를 만들고, 
        이로부터 최종 LTS scale을 계산합니다.
        
        Args:
            outputs (dict): 모델 출력. 'individual_features', 'fusion_features', 'confidences' 필요.
            alpha (float): Fused feature에 대한 가중치.
        
        Returns:
            torch.Tensor: 최종 하이브리드 Feature로부터 계산된 LTS scale [batch_size]
        """
        individual_features = outputs.get('individual_features')
        fusion_features = outputs.get('fusion_features')  # Concatenated feature
        confidences = outputs.get('confidences', {})
        
        if fusion_features is None:
            raise ValueError("LTS Feature Fusion requires 'fusion_features'.")
        if individual_features is None:
            raise ValueError("LTS Feature Fusion requires 'individual_features'.")
        if not confidences:
            raise ValueError("LTS Feature Fusion requires 'confidences'.")
        
        logging.info("  🔥 LTS Feature Fusion:")
        
        # 1. Projection layer 초기화 (첫 실행 시)
        if not self._projection_initialized:
            # individual feature의 dimension 추출
            if isinstance(individual_features, dict):
                sample_feature = list(individual_features.values())[0]
            elif isinstance(individual_features, list):
                sample_feature = individual_features[0]
            else:
                raise ValueError("individual_features must be dict or list")
            
            individual_dim = sample_feature.shape[-1]
            
            # Projection layer 생성: fusion_features dim -> individual_features dim
            self.projection_layer = nn.Linear(fusion_features.shape[-1], individual_dim).to(self.device)
            self._projection_initialized = True
            
            logging.info(f"     - Initialized projection: {fusion_features.shape[-1]} -> {individual_dim}")
        
        # 2. 차원 통일을 위해 Fused feature를 프로젝션
        projected_fused_feature = self.projection_layer(fusion_features)
        
        # 3. 하이브리드 Feature 생성 (가중합)
        hybrid_feature = projected_fused_feature * alpha
        
        logging.info(f"     - Projected fused feature (α={alpha}): {projected_fused_feature.shape}")

        modalities = ['RGB', 'Gyro', 'Acce']
        for i, modality in enumerate(modalities):
            if modality in confidences:
                # individual_features는 dict 또는 list일 수 있음
                if isinstance(individual_features, dict):
                    features = individual_features.get(modality)
                elif isinstance(individual_features, list):
                    features = individual_features[i] if i < len(individual_features) else None
                else:
                    features = None
                
                if features is not None:
                    conf = confidences[modality].unsqueeze(-1)  # 차원 확장: [batch] -> [batch, 1]
                    
                    hybrid_feature = hybrid_feature + (features * conf)
                    
                    logging.info(f"     - Added {modality} feature: {features.shape}, conf={conf.squeeze(-1).mean():.4f}")
                
        # 4. 최종 하이브리드 Feature로부터 LTS scale 계산
        final_scale = self._base_detector._calculate_lts_scale(hybrid_feature)
        
        logging.info(f"     - Hybrid feature: {hybrid_feature.shape}")
        logging.info(f"     - Final scale: mean={final_scale.mean():.4f}")
        
        return final_scale
        
    def _get_fused_logits_for_hybrid(self, outputs):
        """
        모든 하이브리드 모드에서 사용할 Fused Logits를 계산하는 헬퍼 함수
        
        Args:
            outputs: 모델의 forward 출력 딕셔너리
        
        Returns:
            fused_logits: tensor [batch, num_classes]
        """
        main_logits = outputs.get('logits')
        auxiliary_logits = outputs.get('auxiliary_logits', {})
        confidences = outputs.get('confidences', {})
        modalities = ['RGB', 'Gyro', 'Acce']

        if not auxiliary_logits:
            raise ValueError(f"Hybrid mode requires 'auxiliary_logits'.")

        if self.mode in ['hybrid_noweight', 'hybrid_lts_scorefusion', 'hybrid_lts_featurefusion']:
            # LTS 하이브리드 모드들도 Logit은 noweight 방식으로 퓨전
            return self._fuse_logits_noweight(main_logits, auxiliary_logits, modalities)
        elif self.mode == 'hybrid_conf_raw':
            if not confidences: 
                raise ValueError(f"Mode '{self.mode}' requires 'confidences'.")
            return self._fuse_logits_conf_raw(main_logits, auxiliary_logits, confidences, modalities)
        elif self.mode == 'hybrid_conf_normalized':
            if not confidences: 
                raise ValueError(f"Mode '{self.mode}' requires 'confidences'.")
            return self._fuse_logits_conf_normalized(main_logits, auxiliary_logits, confidences, modalities)
        else:
            raise ValueError(f"Unknown mode for logit fusion: {self.mode}")
    
    def _compute_scores_from_logits(self, logits):
        """
        Logits에서 OOD scores 계산 (기존 detector 재사용)
        
        Args:
            logits: [batch, num_classes]
        
        Returns:
            scores: OOD scores (numpy array)
        """
        # 기존 detector의 _compute_scores_from_logits 메서드를 재사용
        return self._base_detector._compute_scores_from_logits(logits)
    
    def _compute_feature_transform_scores(self, outputs):
        """
        Feature transform methods (ReAct, Scale, ASH_S) OOD scores 계산
        
        Args:
            outputs: 모델의 forward 출력 딕셔너리
                - fusion_features: tensor [batch, feature_dim] (필수)
                - logits: Main logits (optional, features로부터 재계산 가능)
        
        Returns:
            scores: OOD scores (numpy array)
        """
        # Features 확인
        if 'fusion_features' not in outputs or outputs['fusion_features'] is None:
            raise ValueError(f"{self.method} requires 'fusion_features' in outputs, but got: {outputs.keys()}")
        
        features = outputs['fusion_features']
        
        # Feature transformation 적용
        if self.method == 'react':
            transformed_features = self._base_detector.react(features)
        elif self.method == 'scale':
            transformed_features = self._base_detector.scale(features)
        elif self.method == 'ash_s':
            transformed_features = self._base_detector.ash_s(features)
        else:
            raise ValueError(f"Unknown feature transform method: {self.method}")
        
        # Transformed features로부터 logits 계산
        # Model의 FC layer 사용
        with torch.no_grad():
            if hasattr(self.model, 'fc'):
                fc_output = self.model.fc(transformed_features)
            elif hasattr(self.model, 'classifier'):
                fc_output = self.model.classifier(transformed_features)
            else:
                raise ValueError("Model has no 'fc' or 'classifier' layer for feature transformation")
            
            # FC layer가 dict를 반환하는 경우 (TBN 등)
            if isinstance(fc_output, dict):
                if 'logits' in fc_output:
                    transformed_logits = fc_output['logits']
                else:
                    raise ValueError(f"FC layer returned dict without 'logits' key: {fc_output.keys()}")
            else:
                transformed_logits = fc_output
        
        # Energy score 계산 (logsumexp)
        energy_scores = torch.logsumexp(transformed_logits, dim=1)
        return energy_scores.cpu().numpy()
    
    def _compute_odin_scores(self, outputs):
        """
        ODIN method OOD scores 계산
        
        Args:
            outputs: 모델의 forward 출력 딕셔너리
                - raw_inputs: tensor (optional, input perturbation용)
                - logits: Main logits (fallback용)
        
        Returns:
            scores: OOD scores (numpy array)
        """
        # Raw inputs가 있으면 ODIN의 full method 사용
        if 'raw_inputs' in outputs and outputs['raw_inputs'] is not None:
            raw_inputs = outputs['raw_inputs']
            return self._base_detector.odin_score(raw_inputs)
        
        # Fallback: logits만 있으면 temperature scaling만 사용
        elif 'logits' in outputs and outputs['logits'] is not None:
            logging.warning(f"ODIN: raw_inputs not provided, using temperature scaling only (no perturbation)")
            return self._base_detector._compute_scores_from_logits(outputs['logits'])
        
        else:
            raise ValueError(f"ODIN requires 'raw_inputs' or 'logits' in outputs, but got: {outputs.keys()}")
    
    def get_config_str(self):
        """설정을 문자열로 반환 (로깅용)"""
        return f"{self.method}_{self.mode}_T{self.temperature}"
    
    def get_method_name(self):
        """
        Config로부터 method name 생성 (로깅 및 결과 저장용)
        
        Returns:
            str: Method name (예: "MSP_Baseline", "Energy_Hybrid_ConfRaw", "ReAct_Baseline")
        """
        # Method 부분 (특수 케이스 처리)
        if self.method == 'msp':
            method_part = 'MSP'
        elif self.method == 'lts':
            method_part = 'LTS'
        elif self.method == 'ash_s':
            method_part = 'ASH_S'
        elif self.method == 'odin':
            method_part = 'ODIN'
        else:
            # 첫 글자만 대문자 (energy → Energy, react → ReAct)
            method_part = self.method.capitalize()
        
        # Mode 부분
        if self.mode == 'baseline':
            mode_part = 'Baseline'
        elif self.mode == 'hybrid_noweight':
            mode_part = 'Hybrid_NoWeight'
        elif self.mode == 'hybrid_conf_raw':
            mode_part = 'Hybrid_ConfRaw'
        elif self.mode == 'hybrid_conf_normalized':
            mode_part = 'Hybrid_ConfNormalized'
        else:
            mode_part = self.mode.capitalize()
        
        return f"{method_part}_{mode_part}"

