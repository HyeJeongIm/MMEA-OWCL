import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
import logging
from utils.basic_ops import ConsensusModule

class AuxiliaryHeadFusionV2_5(nn.Module):
    """
    🔥 Warm-up 지원 Auxiliary Head 기반 융합 모듈 (v2_5) - 빠른 warmup 버전
    
    핵심 개선사항 (vs v2_4):
    1. ✅ Warmup 기간 단축: 5 epoch → 1 epoch
    2. ✅ 더 빠른 동적 가중치 전환
    3. ✅ 학습 초기 불안정성 최소화
    
    Warm-up 메커니즘:
    - 초기 1 epoch: 균등 가중치 (1:1:1) 사용
    - 이후: auxiliary head 기반 동적 가중치 사용
    - 각 task마다 독립적인 warmup (Task 0만 적용)
    
    학습 목표:
    1. 각 auxiliary head가 실제로 좋은 예측을 하도록 학습
    2. 동시에 융합 결과도 최적화
    3. 신뢰도와 실제 성능의 일치성 확보
    4. 🔥 빠른 warmup으로 효율적인 학습
    """
    
    def __init__(self, feature_dim, modality, dropout, num_classes=32, 
                 confidence_method="max_prob", aux_loss_weight=0.5,
                 consensus_type='avg', before_softmax=True, num_segments=8,
                 warmup_epochs=1):
        """
        Args:
            feature_dim: 각 모달리티 특징 차원 (1024)
            modality: 모달리티 리스트 ["RGB", "Gyro", "Acce"]
            dropout: 드롭아웃 확률
            num_classes: 클래스 수 (auxiliary head 출력 차원)
            confidence_method: 신뢰도 계산 방법 ("entropy", "max_prob")
            aux_loss_weight: Auxiliary loss 가중치 (λ) - 기본값 0.5
            consensus_type: TBN consensus 방법 ('avg', 'identity')
            before_softmax: Softmax 적용 여부
            num_segments: TBN segments 수
            warmup_epochs: Warm-up epoch 수 (기본값: 1)
        """
        super().__init__()
        self.modality = modality
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.confidence_method = confidence_method
        self.aux_loss_weight = aux_loss_weight
        
        # TBN consensus 파라미터
        self.consensus_type = consensus_type
        self.before_softmax = before_softmax
        self.num_segments = num_segments
        self.reshape = True
        
        # 모달리티 인덱스 매핑
        self.modality_to_idx = {m: i for i, m in enumerate(self.modality)}
        
        # 🔥 개선된 Warm-up 메커니즘
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0  # 현재 epoch (외부에서 set_epoch()로 주입)
        self.current_task_id = 0
        
        # 🎯 각 모달리티별 auxiliary head
        self.auxiliary_heads = nn.ModuleDict()
        for modality_name in self.modality:
            self.auxiliary_heads[modality_name] = nn.Linear(feature_dim, num_classes)
            normal_(self.auxiliary_heads[modality_name].weight, 0, 0.001)
            constant_(self.auxiliary_heads[modality_name].bias, 0)
        
        # 최종 융합 레이어 (Multi-modal 전용)
        if len(self.modality) <= 1:
            raise ValueError("AuxiliaryHeadFusionV2_4 requires multiple modalities")
        
        input_dim = len(self.modality) * feature_dim
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        normal_(self.fc1.weight, 0, 0.001)
        constant_(self.fc1.bias, 0)
        
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # 디버깅 플래그
        self.first_forward_per_task = {}
        self.epoch_logged = set()  # 이미 로깅된 epoch 추적 (중복 방지)
        
        # TBN consensus
        self.consensus = ConsensusModule(consensus_type)
        
        # Optional softmax
        if not self.before_softmax:
            self.softmax = nn.Softmax(dim=1)

    def set_epoch(self, epoch):
        """
        외부 학습 루프에서 현재 epoch 설정
        
        Args:
            epoch: 현재 epoch (0-based)
        """
        old_epoch = self.current_epoch
        self.current_epoch = epoch
        
        # 🔥 Warmup 전환 시점 로깅 (Task 0에서만, epoch == warmup_epochs일 때)
        if (self.current_task_id == 0 and 
            epoch == self.warmup_epochs and 
            epoch not in self.epoch_logged):
            logging.info(f"")
            logging.info(f"{'='*60}")
            logging.info(f"🎉 WARM-UP COMPLETED!")
            logging.info(f"{'='*60}")
            logging.info(f"   Task: {self.current_task_id}")
            logging.info(f"   Epoch: {old_epoch} → {epoch}")
            logging.info(f"   Weight Mode: Uniform (1:1:1) → Auxiliary-based")
            logging.info(f"   From now on: Using dynamic confidence weights")
            logging.info(f"{'='*60}")
            self.epoch_logged.add(epoch)

    def _pick_features(self, features):
        """features 리스트에서 각 모달리티 특징 추출"""
        f_rgb = features[self.modality_to_idx['RGB']] if 'RGB' in self.modality_to_idx else None
        f_gyro = features[self.modality_to_idx['Gyro']] if 'Gyro' in self.modality_to_idx else None
        f_acce = features[self.modality_to_idx['Acce']] if 'Acce' in self.modality_to_idx else None
        return f_rgb, f_gyro, f_acce
    
    def _is_warmup_phase(self):
        """
        현재가 warm-up 단계인지 확인
        
        Returns:
            bool: True if warm-up phase, False otherwise
        """
        # Task 0에서만 warmup 적용
        if self.current_task_id != 0:
            return False
        return self.current_epoch < self.warmup_epochs

    def _compute_confidence(self, logits):
        """
        Auxiliary head 예측 결과로부터 신뢰도 계산
        
        Args:
            logits: [Batch, num_classes] auxiliary head 출력
            
        Returns:
            confidence: [Batch] 신뢰도 점수 (0~1)
        """
        probs = F.softmax(logits, dim=1)
        
        if self.confidence_method == "entropy":
            eps = 1e-8
            entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
            max_entropy = torch.log(torch.tensor(self.num_classes, device=entropy.device))
            confidence = 1.0 - (entropy / max_entropy)
            
        elif self.confidence_method == "max_prob":
            confidence, _ = torch.max(probs, dim=1)
            
        else:
            confidence = torch.ones(logits.size(0), device=logits.device) / len(self.modality)
        
        return confidence

    def update_task(self, task_id):
        """
        새로운 task 시작 시 호출하여 task ID 업데이트
        
        Args:
            task_id: 새로운 task ID (0, 1, 2, ...)
        """
        old_task_id = self.current_task_id
        self.current_task_id = task_id
        
        if task_id not in self.first_forward_per_task:
            self.first_forward_per_task[task_id] = True
        
        # 🔥 Task 전환 시 epoch 로깅 초기화
        self.epoch_logged.clear()
        
        # 🔥 Task별 warmup 정책
        if task_id == 0:
            self.current_epoch = 0  # Task 0 시작 시 epoch 리셋
            logging.info(f"")
            logging.info(f"{'='*60}")
            logging.info(f"🔥 Task {task_id}: Warm-up ENABLED")
            logging.info(f"{'='*60}")
            logging.info(f"   Duration: epochs 0-{self.warmup_epochs-1} (total {self.warmup_epochs} epochs)")
            logging.info(f"   Weight Mode: Uniform (1:1:1)")
            logging.info(f"   Purpose: Stabilize auxiliary head learning")
            logging.info(f"{'='*60}")
        else:
            logging.info(f"")
            logging.info(f"{'='*60}")
            logging.info(f"🎯 Task {task_id}: Warm-up DISABLED")
            logging.info(f"{'='*60}")
            logging.info(f"   Previous Task: {old_task_id}")
            logging.info(f"   Current Epoch: {self.current_epoch}")
            logging.info(f"   Weight Mode: Auxiliary-based (from start)")
            logging.info(f"   Reason: Only Task 0 uses warm-up")
            logging.info(f"{'='*60}")
    
    def update_auxiliary_heads(self, nb_classes):
        """
        CL에서 새로운 task 시작 시 auxiliary head의 클래스 수 업데이트
        
        Args:
            nb_classes: 새로운 총 클래스 수
        """
        old_num_classes = self.num_classes
        self.num_classes = nb_classes
        
        for modality_name in self.modality:
            old_head = self.auxiliary_heads[modality_name]
            new_head = nn.Linear(self.feature_dim, nb_classes)
            
            # 기존 가중치 보존
            if old_num_classes > 0:
                new_head.weight.data[:old_num_classes] = old_head.weight.data
                new_head.bias.data[:old_num_classes] = old_head.bias.data
            
            # 새로운 클래스 가중치 초기화
            if nb_classes > old_num_classes:
                normal_(new_head.weight.data[old_num_classes:], 0, 0.001)
                constant_(new_head.bias.data[old_num_classes:], 0)
            
            self.auxiliary_heads[modality_name] = new_head
        
        logging.info(f"🎯 Auxiliary heads updated: {old_num_classes} → {nb_classes} classes")

    def _apply_consensus_to_logits(self, aux_logits):
        """
        TBN segments 레벨 logits에 consensus 적용
        
        Args:
            aux_logits: [batch*segments, num_classes]
            
        Returns:
            consensus_logits: [batch, num_classes]
        """
        if self.num_segments <= 1:
            return aux_logits
        
        batch_size = aux_logits.size(0) // self.num_segments
        if aux_logits.size(0) % self.num_segments != 0:
            return aux_logits
        
        base_out = aux_logits
        if not self.before_softmax:
            base_out = self.softmax(base_out)
        
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        
        output = self.consensus(base_out)
        
        if self.consensus_type == 'identity':
            output = output[:, 0, :]
        else:
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
                    aux_logits_segments = self.auxiliary_heads[modality_name](feature)
                    aux_logits = self._apply_consensus_to_logits(aux_logits_segments)
                    confidence = self._compute_confidence(aux_logits)
                    
                    auxiliary_logits[modality_name] = aux_logits
                    confidences[modality_name] = confidence
            
            # 🎯 Warm-up 기반 가중치 선택
            if confidences:
                confidence_tensor = torch.stack(list(confidences.values()), dim=1)
                
                # 🔥 Warm-up 단계 확인
                is_warmup = self._is_warmup_phase()
                
                if is_warmup:
                    # Uniform weights (no scaling, 모든 가중치 = 1)
                    weights = torch.ones_like(confidence_tensor)
                    weight_type = "Uniform (no-scaling)"
                else:
                    # Auxiliary head 기반 동적 가중치
                    confidence_sum = torch.sum(confidence_tensor, dim=1, keepdim=True) + 1e-8
                    weights = confidence_tensor / confidence_sum
                    weight_type = "Auxiliary-based (dynamic)"
                
                # 🔧 디버깅: 첫 forward 또는 주요 전환 시점
                should_debug = (
                    self.first_forward_per_task.get(self.current_task_id, False) or
                    (self.current_task_id == 0 and self.current_epoch in [0, self.warmup_epochs] and 
                     self.current_epoch not in self.epoch_logged)
                )
                
                if should_debug:
                    logging.info(f"")
                    logging.info(f"🔍 Weight Selection (Task {self.current_task_id}, Epoch {self.current_epoch}):")
                    logging.info(f"   🔥 Warm-up: {'Active' if is_warmup else 'Completed'}")
                    logging.info(f"   🎯 Weight Type: {weight_type}")
                    logging.info(f"   📊 Confidence Stats:")
                    for mod_name, conf in confidences.items():
                        conf_np = conf.detach().cpu().numpy()
                        logging.info(f"      {mod_name}: mean={conf_np.mean():.3f}, std={conf_np.std():.3f}, "
                                   f"min={conf_np.min():.3f}, max={conf_np.max():.3f}")
                    
                    # 가중치 통계
                    weights_np = weights[0].detach().cpu().numpy()
                    logging.info(f"   ⚖️  Final Weights (first sample): {weights_np}")
                    
                    if self.current_epoch not in self.epoch_logged:
                        self.epoch_logged.add(self.current_epoch)
                
                # 각 모달리티에 가중치 적용
                weighted_features = []
                weight_list = []
                
                for i, (modality_name, feature) in enumerate(modality_features.items()):
                    if feature is not None and modality_name in confidences:
                        weight = weights[:, list(confidences.keys()).index(modality_name)].unsqueeze(1)
                        
                        if self.num_segments > 1:
                            weight_expanded = weight.repeat_interleave(self.num_segments, dim=0)
                        else:
                            weight_expanded = weight
                        
                        weighted_feature = weight_expanded * feature
                        weighted_features.append(weighted_feature)
                        weight_list.append(weight.squeeze(-1))
                
                # 최종 융합
                x = torch.cat(weighted_features, dim=1)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout_layer(x)
                
            else:
                # Fallback: 균등 가중치
                x = torch.cat(list(modality_features.values()), dim=1)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout_layer(x)
                auxiliary_logits = {}
                weight_list = [torch.ones(x.size(0), device=x.device)] * len(self.modality)
            
        else:
            raise ValueError("AuxiliaryHeadFusionV2_4 requires multiple modalities")
        
        # 🎯 Multi-task Loss 계산
        auxiliary_loss = 0.0
        aux_loss_per_modality = {}
        
        if targets is not None and auxiliary_logits:
            for modality_name, aux_logits in auxiliary_logits.items():
                mod_loss = F.cross_entropy(aux_logits, targets)
                aux_loss_per_modality[modality_name] = mod_loss.item()
                auxiliary_loss += mod_loss
            auxiliary_loss /= len(auxiliary_logits)
        
        # 🔥 디버깅 정보 출력 (첫 forward에서만)
        if self.first_forward_per_task.get(self.current_task_id, False):
            logging.info(f"")
            logging.info(f"🎯 Fusion Module Summary (Task {self.current_task_id}, Epoch {self.current_epoch}):")
            logging.info(f"   🔧 Architecture: Multi-task Learning (Main + Auxiliary)")
            logging.info(f"   📊 Modalities: {list(self.auxiliary_heads.keys())} ({len(self.modality)} total)")
            logging.info(f"   🎯 Confidence Method: {self.confidence_method}")
            logging.info(f"   🔥 Warmup Epochs: {self.warmup_epochs}")
            
            if targets is not None and aux_loss_per_modality:
                logging.info(f"   💰 Loss Analysis:")
                logging.info(f"      Auxiliary Loss (avg): {auxiliary_loss.item():.4f}")
                logging.info(f"      Auxiliary Weight (λ): {self.aux_loss_weight}")
                logging.info(f"      Auxiliary Loss (weighted): {(self.aux_loss_weight * auxiliary_loss).item():.4f}")
                logging.info(f"   📈 Per-Modality Aux Loss:")
                for mod_name, mod_loss in aux_loss_per_modality.items():
                    logging.info(f"      {mod_name}: {mod_loss:.4f}")
            
            logging.info(f"   📚 Task History: {list(self.first_forward_per_task.keys())}")
            logging.info(f"")
            
            self.first_forward_per_task[self.current_task_id] = False
        
        return {
            'features': x,
            'auxiliary_logits': auxiliary_logits,
            'auxiliary_loss': auxiliary_loss,
            'aux_loss_weight': self.aux_loss_weight,
            'modality_weights': torch.stack(weight_list, dim=1).detach() if weight_list else None,
            'confidences': confidences,
            'fusion_type': 'auxiliary_head_v2_4'
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


