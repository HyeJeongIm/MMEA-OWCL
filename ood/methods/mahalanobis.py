import torch
import torch.nn.functional as F
import numpy as np
from .base_ood import BaseOODDetector

class MahalanobisDetector(BaseOODDetector):
    """
    Mahalanobis Distance 기반 Out-of-Distribution Detector
    
    핵심 아이디어:
    - ID 데이터의 penultimate layer features의 분포를 학습
    - 새로운 샘플의 Mahalanobis distance를 계산하여 OOD 판정
    - 거리가 클수록 OOD일 가능성이 높음
    
    방법:
    1. ID 데이터로부터 penultimate features 수집
    2. 클래스별 평균과 공분산 행렬 계산
    3. 새로운 샘플의 Mahalanobis distance 계산
    4. 최소 거리를 OOD 스코어로 사용
    """
    
    def __init__(self, model, device='cuda', magnitude=0.0):
        """
        Args:
            model: 학습된 모델
            device: 디바이스
            magnitude: 입력 perturbation 크기 (0.0 = 사용 안함)
        """
        super().__init__(model, device)
        self.magnitude = magnitude
        self.class_means = None
        self.class_covs = None
        self.global_cov = None
        self.is_fitted = False
        
    def fit(self, dataloader):
        """
        ID 데이터로부터 Mahalanobis distance 파라미터 학습
        
        Args:
            dataloader: ID 데이터 로더
        """
        print("🔧 Mahalanobis detector fitting...")
        
        # Penultimate features와 labels 수집
        features_list = []
        labels_list = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if isinstance(inputs, dict):
                    # Multi-modal 입력 처리
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Penultimate features 추출
                features = self._extract_penultimate_features(inputs)
                features_list.append(features.cpu())
                labels_list.append(targets.cpu())
        
        # 모든 features와 labels 결합
        all_features = torch.cat(features_list, dim=0).numpy()
        all_labels = torch.cat(labels_list, dim=0).numpy()
        
        # 클래스별 평균과 공분산 계산
        self._compute_class_statistics(all_features, all_labels)
        self.is_fitted = True
        print(f"✅ Mahalanobis detector fitted with {len(all_features)} samples")
    
    def _extract_penultimate_features(self, inputs):
        """Penultimate layer features 추출"""
        # 모델의 forward hook을 사용하여 penultimate features 추출
        features = None
        
        def hook_fn(module, input, output):
            nonlocal features
            features = input[0] if isinstance(input, tuple) else input
        
        # Penultimate layer에 hook 등록 (일반적으로 마지막 FC layer 직전)
        penultimate_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'classifier' in name.lower():
                # 마지막 classifier 직전의 layer 찾기
                penultimate_layer = module
                break
        
        if penultimate_layer is None:
            # Fallback: 마지막 layer 사용
            penultimate_layer = list(self.model.modules())[-1]
        
        hook = penultimate_layer.register_forward_hook(hook_fn)
        
        try:
            _ = self.model(inputs)
        finally:
            hook.remove()
        
        if features is None:
            # Fallback: logits 사용
            outputs = self.model(inputs)
            if isinstance(outputs, dict):
                features = outputs.get('logits', torch.zeros(1, 10))
            else:
                features = outputs
        
        return features
    
    def _compute_class_statistics(self, features, labels):
        """클래스별 통계량 계산"""
        unique_labels = np.unique(labels)
        self.class_means = {}
        self.class_covs = {}
        
        # 클래스별 평균과 공분산 계산
        for label in unique_labels:
            class_mask = labels == label
            class_features = features[class_mask]
            
            if len(class_features) > 1:
                self.class_means[label] = np.mean(class_features, axis=0)
                self.class_covs[label] = np.cov(class_features.T)
            else:
                # 샘플이 1개인 경우
                self.class_means[label] = class_features[0]
                self.class_covs[label] = np.eye(features.shape[1]) * 1e-6
        
        # Global covariance 계산 (모든 클래스 통합)
        self.global_cov = np.cov(features.T)
        
        # Numerical stability를 위한 regularization
        reg_term = 1e-6
        self.global_cov += reg_term * np.eye(self.global_cov.shape[0])
        
        for label in self.class_covs:
            self.class_covs[label] += reg_term * np.eye(self.class_covs[label].shape[0])
    
    def _compute_scores_from_logits(self, logits):
        """
        Logits에서 Mahalanobis distance 계산
        (Fallback method - 실제로는 penultimate features 필요)
        """
        if not self.is_fitted:
            # Fitting이 안된 경우 logits의 norm 사용
            return torch.norm(logits, dim=1).cpu().numpy()
        
        # Logits를 features로 간주하고 계산
        features = logits.cpu().numpy()
        scores = []
        
        for feature in features:
            min_distance = float('inf')
            
            # 각 클래스와의 Mahalanobis distance 계산
            for label, mean in self.class_means.items():
                diff = feature - mean
                cov = self.class_covs[label]
                
                try:
                    # Mahalanobis distance 계산
                    inv_cov = np.linalg.inv(cov)
                    distance = np.sqrt(diff.T @ inv_cov @ diff)
                    min_distance = min(min_distance, distance)
                except np.linalg.LinAlgError:
                    # 역행렬 계산 실패시 Euclidean distance 사용
                    distance = np.linalg.norm(diff)
                    min_distance = min(min_distance, distance)
            
            scores.append(min_distance)
        
        return np.array(scores)
    
    def compute_scores_from_outputs(self, outputs):
        """
        모델 출력에서 Mahalanobis distance 계산
        """
        if not self.is_fitted:
            # Fitting이 안된 경우 logits 사용
            if 'logits' in outputs:
                return self._compute_scores_from_logits(outputs['logits'])
            return np.zeros(1)
        
        # Penultimate features 추출 (실제 구현에서는 hook 사용)
        if 'penultimate_features' in outputs:
            features = outputs['penultimate_features'].cpu().numpy()
        else:
            # Fallback: logits 사용
            features = outputs['logits'].cpu().numpy()
        
        scores = []
        for feature in features:
            min_distance = float('inf')
            
            # 각 클래스와의 Mahalanobis distance 계산
            for label, mean in self.class_means.items():
                diff = feature - mean
                cov = self.class_covs[label]
                
                try:
                    # Mahalanobis distance 계산
                    inv_cov = np.linalg.inv(cov)
                    distance = np.sqrt(diff.T @ inv_cov @ diff)
                    min_distance = min(min_distance, distance)
                except np.linalg.LinAlgError:
                    # 역행렬 계산 실패시 Euclidean distance 사용
                    distance = np.linalg.norm(diff)
                    min_distance = min(min_distance, distance)
            
            scores.append(min_distance)
        
        return np.array(scores)
