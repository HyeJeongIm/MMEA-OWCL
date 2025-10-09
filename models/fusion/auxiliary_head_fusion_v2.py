import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
import logging
from utils.basic_ops import ConsensusModule

class AuxiliaryHeadFusionV2(nn.Module):
    """
    🎯 개선된 Auxiliary Head 기반 융합 모듈 (v2)
    
    핵심 개선사항:
    1. Main task loss + Auxiliary loss 결합 (Multi-task Learning)
    2. 실제 예측 성능과 연결된 신뢰도 계산
    3. 해석 가능한 모달리티별 기여도
    
    차이점 (vs v1):
    - v1: Main loss만 사용, auxiliary head는 간접 학습
    - v2: Main + Auxiliary loss 결합, 직접적인 multi-task 학습
    
    학습 목표:
    1. 각 auxiliary head가 실제로 좋은 예측을 하도록 학습
    2. 동시에 융합 결과도 최적화
    3. 신뢰도와 실제 성능의 일치성 확보
    """
    
    def __init__(self, feature_dim, modality, dropout, num_classes=32, 
                 confidence_method="max_prob", aux_loss_weight=0.5,
                 consensus_type='avg', before_softmax=True, num_segments=8):
        """
        Args:
            feature_dim: 각 모달리티 특징 차원 (1024)
            modality: 모달리티 리스트 ["RGB", "Gyro", "Acce"]
            dropout: 드롭아웃 확률
            num_classes: 클래스 수 (auxiliary head 출력 차원)
            confidence_method: 신뢰도 계산 방법 ("entropy", "max_prob")
            aux_loss_weight: Auxiliary loss 가중치 (λ) - 기본값 0.5
            consensus_type: TBN consensus 방법 ('avg', 'identity') - TBNClassification과 동일
            before_softmax: Softmax 적용 여부 - TBNClassification과 동일
            num_segments: TBN segments 수 - TBNClassification과 동일
        """
        super().__init__()
        self.modality = modality
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.confidence_method = confidence_method
        self.aux_loss_weight = aux_loss_weight  # λ 파라미터
        
        # TBN consensus 파라미터 (TBNClassification과 동일)
        self.consensus_type = consensus_type
        self.before_softmax = before_softmax
        self.num_segments = num_segments
        self.reshape = True  # TBNClassification과 동일
        
        # 모달리티 인덱스 매핑
        self.modality_to_idx = {m: i for i, m in enumerate(self.modality)}
        
        # 🎯 각 모달리티별 auxiliary head
        self.auxiliary_heads = nn.ModuleDict()
        for modality_name in self.modality:
            self.auxiliary_heads[modality_name] = nn.Linear(feature_dim, num_classes)
            # 가중치 초기화
            normal_(self.auxiliary_heads[modality_name].weight, 0, 0.001)
            constant_(self.auxiliary_heads[modality_name].bias, 0)
        
        # 최종 융합 레이어 (Multi-modal 전용)
        if len(self.modality) <= 1:
            raise ValueError("AuxiliaryHeadFusionV2 requires multiple modalities")
        
        input_dim = len(self.modality) * feature_dim
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        normal_(self.fc1.weight, 0, 0.001)
        constant_(self.fc1.bias, 0)
        
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # 디버깅 플래그 (task별 관리)
        self.first_forward_per_task = {}  # task_id별로 첫 번째 forward 추적
        self.current_task_id = 0  # 현재 task ID
        
        # TBN segments 처리를 위한 consensus (TBNClassification과 동일)
        self.consensus = ConsensusModule(consensus_type)
        
        # Optional softmax (TBNClassification과 동일)
        if not self.before_softmax:
            self.softmax = nn.Softmax(dim=1)  # Fix deprecation warning

    def _pick_features(self, features):
        """features 리스트에서 각 모달리티 특징 추출"""
        f_rgb = features[self.modality_to_idx['RGB']] if 'RGB' in self.modality_to_idx else None
        f_gyro = features[self.modality_to_idx['Gyro']] if 'Gyro' in self.modality_to_idx else None
        f_acce = features[self.modality_to_idx['Acce']] if 'Acce' in self.modality_to_idx else None
        return f_rgb, f_gyro, f_acce

    def _compute_confidence(self, logits):
        """
        Auxiliary head 예측 결과로부터 신뢰도 계산
        
        Args:
            logits: [Batch, num_classes] auxiliary head 출력
            
        Returns:
            confidence: [Batch] 신뢰도 점수 (0~1)
        """
        probs = F.softmax(logits, dim=1)  # [Batch, num_classes]
        
        if self.confidence_method == "entropy":
            # 엔트로피 기반: 낮은 엔트로피 = 높은 신뢰도
            eps = 1e-8
            entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)  # [Batch]
            max_entropy = torch.log(torch.tensor(self.num_classes, device=entropy.device))
            confidence = 1.0 - (entropy / max_entropy)  # 엔트로피 낮을수록 신뢰도 높음
            
        elif self.confidence_method == "max_prob":
            # 최대 확률 기반: 높은 최대 확률 = 높은 신뢰도
            confidence, _ = torch.max(probs, dim=1)  # [Batch]
            
        else:
            # 기본값: 균등 신뢰도
            confidence = torch.ones(logits.size(0), device=logits.device) / len(self.modality)
        
        return confidence

    def update_task(self, task_id):
        """
        새로운 task 시작 시 호출하여 task ID 업데이트
        
        Args:
            task_id: 새로운 task ID (0, 1, 2, ...)
        """
        self.current_task_id = task_id
        if task_id not in self.first_forward_per_task:
            self.first_forward_per_task[task_id] = True
    
    def update_auxiliary_heads(self, nb_classes):
        """
        CL에서 새로운 task 시작 시 auxiliary head의 클래스 수 업데이트
        
        Args:
            nb_classes: 새로운 총 클래스 수
        """
        old_num_classes = self.num_classes
        self.num_classes = nb_classes
        
        # 각 모달리티별 auxiliary head 업데이트
        for modality_name in self.modality:
            old_head = self.auxiliary_heads[modality_name]
            
            # 새로운 auxiliary head 생성
            new_head = nn.Linear(self.feature_dim, nb_classes)
            
            # 기존 가중치 보존 (이전 클래스들)
            if old_num_classes > 0:
                new_head.weight.data[:old_num_classes] = old_head.weight.data
                new_head.bias.data[:old_num_classes] = old_head.bias.data
            
            # 새로운 클래스들의 가중치 초기화
            if nb_classes > old_num_classes:
                from torch.nn.init import normal_, constant_
                normal_(new_head.weight.data[old_num_classes:], 0, 0.001)
                constant_(new_head.bias.data[old_num_classes:], 0)
            
            # 업데이트
            self.auxiliary_heads[modality_name] = new_head
        
        logging.info(f"🎯 Auxiliary heads updated: {old_num_classes} → {nb_classes} classes")

    def _apply_consensus_to_logits(self, aux_logits):
        """
        TBN segments 레벨 logits에 consensus 적용 (TBNClassification과 완전히 동일한 패턴)
        
        Args:
            aux_logits: [batch*segments, num_classes]
            
        Returns:
            consensus_logits: [batch, num_classes]
        """
        if self.num_segments <= 1:
            # Segments가 1개면 그대로 반환
            return aux_logits
        
        batch_size = aux_logits.size(0) // self.num_segments
        if aux_logits.size(0) % self.num_segments != 0:
            # Segments 수가 맞지 않으면 그대로 반환 (안전장치)
            return aux_logits
        
        # TBNClassification과 완전히 동일한 패턴
        # Step 1: Optional softmax (before_softmax=False일 때)
        base_out = aux_logits
        if not self.before_softmax:
            base_out = self.softmax(base_out)
        
        # Step 2: Reshape [batch*segments, num_classes] → [batch, segments, num_classes]
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        
        # Step 3: Consensus 적용 [batch, segments, num_classes] → [batch, 1, num_classes]
        output = self.consensus(base_out)
        
        # Step 4: Squeeze [batch, 1, num_classes] → [batch, num_classes]
        # Identity consensus는 차원이 다를 수 있으므로 안전하게 처리
        if self.consensus_type == 'identity':
            # Identity: [batch, segments, num_classes] → [batch, segments, num_classes]
            # 첫 번째 segment만 사용 (TBN에서 일반적인 방법)
            output = output[:, 0, :]  # [batch, num_classes]
        else:
            # Avg: [batch, 1, num_classes] → [batch, num_classes]
            output = output.squeeze(1)
        
        return output


    def forward(self, features, targets=None):
        """
        Forward pass: Multi-task Learning (Main + Auxiliary)
        
        Args:
            features: List[Tensor] - [f_rgb, f_gyro, f_acce]
            targets: 정답 레이블 (auxiliary loss 계산용, 필수!)
            
        Returns:
            dict: 융합된 특징 + auxiliary 정보 + loss 정보
        """
        if len(self.modality) > 1:
            # 각 모달리티 특징 분리
            f_rgb, f_gyro, f_acce = self._pick_features(features)
            modality_features = {'RGB': f_rgb, 'Gyro': f_gyro, 'Acce': f_acce}
            
            # 🎯 각 모달리티별 auxiliary head 예측 및 신뢰도 계산
            auxiliary_logits = {}
            confidences = {}
            
            for modality_name, feature in modality_features.items():
                if feature is not None and modality_name in self.auxiliary_heads:
                    # TBNClassification 스타일: segments 레벨에서 예측 후 consensus 적용
                    aux_logits_segments = self.auxiliary_heads[modality_name](feature)  # [64, num_classes]
                    aux_logits = self._apply_consensus_to_logits(aux_logits_segments)   # [8, num_classes]
                    confidence = self._compute_confidence(aux_logits)  # [8]
                    
                    # 🔧 디버깅: Task별 크기 변화 추적
                    if self.first_forward_per_task.get(self.current_task_id, False):
                        print(f"🔍 {modality_name} Debug (Task {self.current_task_id}):")
                        print(f"   feature: {feature.shape}")
                        print(f"   aux_logits_segments: {aux_logits_segments.shape}")
                        print(f"   aux_logits (after consensus): {aux_logits.shape}")
                        print(f"   confidence: {confidence.shape}")
                        print(f"   confidence values: {confidence.detach().cpu().numpy()}")
                    
                    auxiliary_logits[modality_name] = aux_logits
                    confidences[modality_name] = confidence
            
            # 🎯 신뢰도 기반 가중치 정규화
            if confidences:
                confidence_tensor = torch.stack(list(confidences.values()), dim=1)  # [Batch, num_modalities]
                
                # 단순 합 정규화: 각 신뢰도를 전체 합으로 나누기
                confidence_sum = torch.sum(confidence_tensor, dim=1, keepdim=True) + 1e-8  # [Batch, 1]
                weights = confidence_tensor / confidence_sum  # [Batch, num_modalities]
                
                # 🔧 디버깅: 가중치 정규화 과정
                if self.first_forward_per_task.get(self.current_task_id, False):
                    print(f"🔍 Weight Normalization Debug (Task {self.current_task_id}):")
                    print(f"   confidence_tensor: {confidence_tensor.shape}")
                    print(f"   weights (normalized): {weights.shape}")
                    print(f"   weights values: {weights.detach().cpu().numpy()}")
                
                # 각 모달리티에 가중치 적용
                weighted_features = []
                weight_list = []
                
                for i, (modality_name, feature) in enumerate(modality_features.items()):
                    if feature is not None and modality_name in confidences:
                        # TBN 방식: segments 레벨에서 가중치 적용
                        weight = weights[:, list(confidences.keys()).index(modality_name)].unsqueeze(1)  # [Batch, 1]
                        
                        # Weight를 segments 레벨로 확장: [8, 1] → [64, 1]
                        if self.num_segments > 1:
                            weight_expanded = weight.repeat_interleave(self.num_segments, dim=0)  # [64, 1]
                        else:
                            weight_expanded = weight
                        
                        # 🔧 디버깅: Weight 확장 과정
                        if self.first_forward_per_task.get(self.current_task_id, False) and i == 0:  # 첫 번째 모달리티만
                            print(f"🔍 Weight Expansion Debug (Task {self.current_task_id}):")
                            print(f"   weight (video-level): {weight.shape} = {weight.squeeze().detach().cpu().numpy()}")
                            print(f"   weight_expanded (segment-level): {weight_expanded.shape}")
                            print(f"   first 16 expanded values: {weight_expanded.squeeze()[:16].detach().cpu().numpy()}")
                        
                        # Segments 레벨에서 가중치 적용
                        weighted_feature = weight_expanded * feature  # [64, 1024]
                        weighted_features.append(weighted_feature)
                        weight_list.append(weight.squeeze(-1))  # Only squeeze last dimension
                
                # 최종 융합: TBN 방식으로 segments → video 레벨
                x = torch.cat(weighted_features, dim=1)  # [64, 3072] (3 modalities × 1024)
                
                # FC layer 적용 후 consensus로 집계
                x = self.fc1(x)  # [64, 512]
                x = self.relu(x)
                
                # 🎯 Main feature는 segments 레벨 그대로 유지 (TBNClassification이 consensus 처리)
                x = self.dropout_layer(x)  # [64, 512] 그대로 출력
                
                # 🔧 디버깅: 최종 출력 크기 확인
                if self.first_forward_per_task.get(self.current_task_id, False):
                    print(f"🔍 Final Fusion Output Debug (Task {self.current_task_id}):")
                    print(f"   Main features (segments level): {x.shape}")
                    print(f"   → TBNClassification will apply consensus")
                
            else:
                # Fallback: 균등 가중치 (auxiliary head가 없는 경우)
                x = torch.cat(list(modality_features.values()), dim=1)  # [64, 3072]
                x = self.fc1(x)  # [64, 512]
                x = self.relu(x)
                
                # 🎯 Fallback도 segments 레벨 그대로 유지
                x = self.dropout_layer(x)  # [64, 512] 그대로 출력
                auxiliary_logits = {}
                weight_list = [torch.ones(x.size(0), device=x.device) / len(self.modality)] * len(self.modality)
            
        # Single modality는 지원하지 않음 (Multi-modal fusion 전용)
        else:
            raise ValueError("AuxiliaryHeadFusionV2 requires multiple modalities (len(modality) > 1)")
        
        # 🎯 Multi-task Loss 계산
        auxiliary_loss = 0.0
        if targets is not None and auxiliary_logits:
            for modality_name, aux_logits in auxiliary_logits.items():
                auxiliary_loss += F.cross_entropy(aux_logits, targets)
            auxiliary_loss /= len(auxiliary_logits)  # 평균
        
        # 디버깅 정보 출력 (각 task의 첫 번째 forward에서만)
        if self.first_forward_per_task.get(self.current_task_id, False):
            print(f"🎯 AuxiliaryHeadFusionV2 Debug (Task {self.current_task_id}):")
            print(f"   Modality count: {len(self.modality)}")
            print(f"   Confidence method: {self.confidence_method}")
            print(f"   Auxiliary loss weight (λ): {self.aux_loss_weight}")
            print(f"   Auxiliary heads: {list(self.auxiliary_heads.keys())}")
            if len(weight_list) > 0:
                print(f"   Weight ranges: [{weight_list[0].min().item():.3f}, {weight_list[0].max().item():.3f}]")
            if targets is not None:
                print(f"   Auxiliary loss: {auxiliary_loss.item():.4f}")
            print(f"   Architecture: Multi-task Learning (Main + Auxiliary)")
            print(f"   Task transition: Previous tasks {list(self.first_forward_per_task.keys())}")
            self.first_forward_per_task[self.current_task_id] = False
        
        return {
            'features': x,
            'auxiliary_logits': auxiliary_logits,
            'auxiliary_loss': auxiliary_loss,
            'aux_loss_weight': self.aux_loss_weight,
            'modality_weights': torch.stack(weight_list, dim=1).detach() if weight_list else None,
            'confidences': confidences,
            'fusion_type': 'auxiliary_head_v2'
        }
    
    def compute_total_loss(self, main_loss, auxiliary_loss=None):
        """
        Main loss와 Auxiliary loss를 결합한 총 손실 계산
        
        Args:
            main_loss: 주 작업 손실 (예: CrossEntropy)
            auxiliary_loss: 보조 작업 손실 (forward에서 반환됨)
            
        Returns:
            total_loss: 결합된 총 손실
        """
        if auxiliary_loss is None or self.aux_loss_weight == 0:
            return main_loss
        
        total_loss = main_loss + self.aux_loss_weight * auxiliary_loss
        
        return total_loss
    
    def get_loss_breakdown(self, main_loss, auxiliary_loss=None):
        """
        손실 구성 요소별 분석 (디버깅용)
        
        Returns:
            dict: 손실 구성 요소 정보
        """
        if auxiliary_loss is None:
            auxiliary_loss = 0.0
            
        total_loss = self.compute_total_loss(main_loss, auxiliary_loss)
        
        return {
            'main_loss': main_loss.item() if torch.is_tensor(main_loss) else main_loss,
            'auxiliary_loss': auxiliary_loss.item() if torch.is_tensor(auxiliary_loss) else auxiliary_loss,
            'aux_loss_weight': self.aux_loss_weight,
            'weighted_aux_loss': (self.aux_loss_weight * auxiliary_loss).item() if torch.is_tensor(auxiliary_loss) else 0.0,
            'total_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            'aux_contribution_ratio': (self.aux_loss_weight * auxiliary_loss / total_loss).item() if torch.is_tensor(total_loss) and total_loss != 0 else 0.0
        }
