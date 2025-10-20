import torch
import torch.nn.functional as F
import numpy as np
from .base_ood import BaseOODDetector

class HellingerDiscrepancyDetector(BaseOODDetector):
    """
    🎯 Auxiliary Logits 간의 헬링거 거리(Hellinger Distance)를 이용한 불일치 측정
    
    핵심 아이디어:
    - 각 모달리티의 auxiliary_logits를 개별적으로 확률 분포(Softmax)로 변환합니다.
    - 모달리티 쌍(pair)마다 헬링거 거리를 계산하여 예측 분포가 얼마나 다른지(불일치)를 측정합니다.
    - 계산된 평균 거리에 음수(-)를 취해 최종 점수를 만듭니다.
      - 불일치가 클수록(OOD일수록) 거리가 커지고, 최종 점수는 낮아집니다.
      - 불일치가 작을수록(ID일수록) 거리가 작아지고, 최종 점수는 높아집니다.
    
    평가 코드와의 호환성:
    - 점수가 높을수록 ID, 낮을수록 OOD로 판정하는 평가 방식과 완벽하게 호환됩니다.
    """
    
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)

    def _hellinger_distance(self, p, q):
        """두 확률 분포 텐서(p, q) 간의 헬링거 거리를 배치 단위로 계산합니다."""
        # torch.norm을 사용하여 L2 norm을 효율적으로 계산
        # 공식: (1/sqrt(2)) * ||sqrt(p) - sqrt(q)||_2
        distance = torch.norm(torch.sqrt(p) - torch.sqrt(q), p=2, dim=1) / np.sqrt(2)
        return distance

    def compute_scores_from_outputs(self, outputs):
        """
        모달리티 간 예측 불일치를 계산하여 OOD 점수를 반환합니다.
        """
        auxiliary_logits = outputs.get('auxiliary_logits', {})
        
        # 비교할 모달리티가 2개 미만이면 OOD 탐지가 불가능하므로 기본값 반환
        if not auxiliary_logits or len(auxiliary_logits) < 2:
            # 배치 크기에 맞는 0점 배열을 반환하는 것이 더 안전할 수 있습니다.
            # 예시: batch_size = next(iter(outputs.values())).size(0)
            return np.zeros(1)

        # STEP 1: 각 모달리티의 logits을 확률 분포로 변환
        probabilities = {
            mod: F.softmax(logits, dim=1) 
            for mod, logits in auxiliary_logits.items()
        }
        
        # 모달리티 순서를 고정하여 일관성 유지
        modalities = ['RGB', 'Gyro', 'Acce']
        probs_list = [probabilities[mod] for mod in modalities if mod in probabilities]
        
        if len(probs_list) < 2:
             return np.zeros(1)

        # STEP 2: 모든 모달리티 쌍(pair)에 대해 헬링거 거리 계산
        discrepancies = []
        # (RGB, Gyro), (RGB, Acce), (Gyro, Acce) 쌍에 대해 거리를 계산
        for i in range(len(probs_list)):
            for j in range(i + 1, len(probs_list)):
                dist = self._hellinger_distance(probs_list[i], probs_list[j])
                discrepancies.append(dist)
        
        # [num_pairs, batch_size] 형태의 텐서들을 stack
        discrepancies_stacked = torch.stack(discrepancies, dim=0)
        
        # STEP 3: 거리들의 평균을 내어 최종 불일치 점수 계산
        # 각 샘플의 평균 불일치 정도를 나타내는 [batch_size] 텐서
        mean_discrepancy = torch.mean(discrepancies_stacked, dim=0)
        
        # STEP 4: 점수 부호 반전 (가장 중요!)
        # 평가 코드(higher score = ID)에 맞추기 위해 불일치 점수에 음수를 취함
        ood_scores = -mean_discrepancy
        
        return ood_scores.cpu().numpy()

    def _compute_scores_from_logits(self, logits):
        """
        BaseOODDetector의 추상 메서드 구현
        단일 logits 입력에 대한 호환성을 위한 래퍼
        """
        # HellingerDiscrepancyDetector는 auxiliary_logits이 필요하므로
        # 단일 logits 입력에 대해서는 기본값 반환
        if isinstance(logits, torch.Tensor):
            batch_size = logits.size(0)
        else:
            batch_size = 1
        return np.zeros(batch_size)
