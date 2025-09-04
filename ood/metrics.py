import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, average_precision_score, confusion_matrix
import logging


def compute_fpr95(id_scores, ood_scores):
    """Compute FPR at 95% TPR"""
    try:
        # Create labels: 1 for ID, 0 for OOD
        y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        y_scores = np.concatenate([id_scores, ood_scores])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        fpr95_idx = np.argmax(tpr >= 0.95)
        fpr95 = fpr[fpr95_idx] * 100 if fpr95_idx < len(fpr) else 100.0
        return fpr95
    except:
        return 100.0


def compute_auroc(id_scores, ood_scores):
    """Compute AUROC"""
    try:
        y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        y_scores = np.concatenate([id_scores, ood_scores])
        return roc_auc_score(y_true, y_scores) * 100
    except:
        return 50.0


def compute_aupr_id(id_scores, ood_scores):
    """
    Compute AUPR with ID as positive class (percent).
    Higher scores indicate better ID detection.
    """
    try:
        y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])  # ID=1, OOD=0
        y_scores = np.concatenate([id_scores, ood_scores])
        return average_precision_score(y_true, y_scores) * 100
    except Exception:
        return 0.0


def compute_aupr_ood(id_scores, ood_scores):
    """
    Compute AUPR with OOD as positive class (percent).
    Since higher scores = ID, lower scores = OOD, we need to flip scores.
    """
    try:
        y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])  # ID=0, OOD=1
        # Flip scores: since high score = ID, we need -score for OOD detection
        y_scores = np.concatenate([-id_scores, -ood_scores])  
        return average_precision_score(y_true, y_scores) * 100
    except Exception:
        return 0.0


def compute_aupr(id_scores, ood_scores):
    """Backward compatibility - same as compute_aupr_id"""
    return compute_aupr_id(id_scores, ood_scores)
    
    
def confusion_matrix_at_fpr95(id_scores, ood_scores):
    """
    FPR@95 TPR 기준 임계값에서 혼동행렬 및 관련 지표 계산
    반환:
      cm_dict = {'tp','fp','tn','fn','threshold','tpr','fpr','precision','recall','f1'}
    """
    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fpr95_idx = np.argmax(tpr >= 0.95)

    # threshold 선택 (sklearn은 thresholds와 fpr/tpr 길이가 동일)
    if fpr95_idx >= len(thresholds):
        thr = thresholds[-1]
        tpr_at = tpr[-1]
        fpr_at = fpr[-1]
    else:
        thr = thresholds[fpr95_idx]
        tpr_at = tpr[fpr95_idx]
        fpr_at = fpr[fpr95_idx]

    # threshold로 예측 생성 (ID=1, OOD=0)
    y_pred = (y_scores >= thr).astype(int)

    # Confusion matrix (순서: labels=[1,0])
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    # cm =
    # [[TP, FN],
    #  [FP, TN]]
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]

    # 유도 지표
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # 양성 예측 중 진짜 ID
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0      # TPR (ID 재현율)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'threshold': float(thr),
        'tpr': float(tpr_at),
        'fpr': float(fpr_at),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }
    

def confusion_matrix_at_youdenJ(id_scores, ood_scores, larger_is_id: bool = True):
    """
    Youden’s J (J = TPR - FPR) 최대가 되는 임계값에서 혼동행렬 및 지표 계산
    반환:
      cm_dict = {'tp','fp','tn','fn','threshold','tpr','fpr','precision','recall','f1','youdenJ'}
    Args:
      id_scores: ID(양성=1) 샘플의 점수 배열
      ood_scores: OOD(음성=0) 샘플의 점수 배열
      larger_is_id: 점수가 클수록 ID면 True (현재 구현은 True)
    """
    # 라벨/점수 결합
    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))]).astype(int)
    y_scores = np.concatenate([id_scores, ood_scores]).astype(float)

    # 점수 방향 보정 (필요 시 부호 반전)
    if not larger_is_id:
        y_scores = -y_scores

    # ROC 곡선
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # 주의: len(thresholds) == len(tpr) - 1 (첫 점 (0,0)은 threshold 없음)

    # Youden’s J = TPR - FPR (threshold 있는 구간만 고려: index 1..end)
    j_all = tpr - fpr
    if len(thresholds) == 0:
        # 모든 점수가 동일 등 예외 상황
        thr = float('inf')
        tpr_at = float(tpr[-1])
        fpr_at = float(fpr[-1])
        youdenJ = float(j_all[-1])
    else:
        idx = np.argmax(j_all[1:]) + 1  # 1..end 중 최댓값의 실제 인덱스
        thr = float(thresholds[idx - 1])  # thresholds[i] ↔ (tpr,fpr)[i+1]
        tpr_at = float(tpr[idx])
        fpr_at = float(fpr[idx])
        youdenJ = float(j_all[idx])

    # 임계값 적용 (ID=1)
    y_pred = (y_scores >= thr).astype(int)

    # 혼동행렬 (labels=[1,0] 순서 고정)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn = int(cm[0, 0]), int(cm[0, 1])
    fp, tn = int(cm[1, 0]), int(cm[1, 1])

    # 지표
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'threshold': thr,
        'tpr': tpr_at,
        'fpr': fpr_at,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'youdenJ': youdenJ,
    }

    
def compute_ood_metrics(id_scores, ood_scores, method_name="OOD"):
    """Compute comprehensive OOD metrics"""
    try:
        auroc = compute_auroc(id_scores, ood_scores)
        aupr_id = compute_aupr_id(id_scores, ood_scores)    # ID as positive
        aupr_ood = compute_aupr_ood(id_scores, ood_scores)  # OOD as positive
        cm_fpr95 = confusion_matrix_at_fpr95(id_scores, ood_scores)  # FPR95 기준
        cm_youden = confusion_matrix_at_youdenJ(id_scores, ood_scores)  # Youden's J 기준

        return {
            'method': method_name,
            'auroc': auroc,
            'aupr_id': aupr_id,     # ID detection performance
            'aupr_ood': aupr_ood,   # OOD detection performance
            'id_samples': len(id_scores),
            'ood_samples': len(ood_scores),
            'confusion_fpr95': cm_fpr95,      # FPR95 기준 confusion matrix
            'confusion_youden': cm_youden     # Youden's J 기준 confusion matrix
        }
    except Exception as e:
        return {
            'method': method_name,
            'error': str(e)
        }


def compute_threshold_accuracy(id_scores, ood_scores, threshold):
    """Compute accuracy with given threshold"""
    try:
        # ID predictions: score >= threshold → 1 (ID), score < threshold → 0 (OOD)
        id_predictions = (np.array(id_scores) >= threshold).astype(int)
        ood_predictions = (np.array(ood_scores) >= threshold).astype(int)
        
        # True labels
        id_labels = np.ones(len(id_scores))  # All ID samples should be 1
        ood_labels = np.zeros(len(ood_scores))  # All OOD samples should be 0
        
        # Compute accuracy
        all_predictions = np.concatenate([id_predictions, ood_predictions])
        all_labels = np.concatenate([id_labels, ood_labels])
        
        accuracy = accuracy_score(all_labels, all_predictions)
        return accuracy
    except:
        return 0.0