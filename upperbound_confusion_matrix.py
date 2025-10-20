#!/usr/bin/env python3
"""
Upperbound 모델 Confusion Matrix + Difference Matrix 통합 생성 스크립트
- 모든 모달리티에 대한 confusion matrix 생성
- 모달리티 간 차이 분석 confusion matrix 생성
- 통합 결과 저장 및 성능 분석
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
from tqdm import tqdm

# 프로젝트 루트를 Python path에 추가
sys.path.append('/workspace/MMEA-OWCL')

from models.model_factory import get_model
from utils.utils import set_random_seed, set_device
from dataloader.data_manager import TBNDataManager, TSNDataManager

# ================= 클래스 순서/이름 =================
COMMON_CLASS_ORDER = [
    26, 14, 23, 4, 11, 25, 31, 10,
    29, 5, 6, 9, 17, 22, 2, 19,
    13, 1, 21, 16, 8, 3, 27, 28,
    15, 30, 0, 7, 12, 18, 20, 24
]

ORIGINAL_CLASS_NAMES = {
    0: 'upstairs', 1: 'downstairs', 2: 'drinking', 3: 'fall', 4: 'reading',
    5: 'sweep_floor', 6: 'cut_fruits', 7: 'mop_floor', 8: 'writing', 9: 'wipe_table',
    10: 'wash_hand', 11: 'standing', 12: 'play_phone', 13: 'type_pc', 14: 'eating',
    15: 'cooking', 16: 'pick_up_phone', 17: 'drop_trush', 18: 'fold_clothes', 19: 'walking',
    20: 'play_card', 21: 'brush_teeth', 22: 'wash_dish', 23: 'moving_sth', 24: 'type_phone',
    25: 'chat', 26: 'open_close_door', 27: 'ride_bike', 28: 'sit_stand', 29: 'take_drop_sth',
    30: 'shopping', 31: 'watch_TV'
}

# ================= Config 유틸 =================
def parse_modality_from_folder(folder_name):
    """폴더명에서 모달리티 추출"""
    folder_name = folder_name.lower()
    if 'rgbgyroacce' in folder_name:
        return ['RGB', 'Gyro', 'Acce']
    elif 'rgbgyro' in folder_name:
        return ['RGB', 'Gyro']
    elif 'rgbacce' in folder_name:
        return ['RGB', 'Acce']
    elif 'gyroacce' in folder_name:
        return ['Gyro', 'Acce']
    elif 'rgb' in folder_name:
        return ['RGB']
    elif 'gyro' in folder_name:
        return ['Gyro']
    elif 'acce' in folder_name:
        return ['Acce']
    else:
        return ['RGB']

def load_and_modify_config(base_config_path, weights_path, folder_name):
    """기존 JSON 설정 불러와 eval 모드로 수정"""
    print(f"📋 기본 설정 로드: {base_config_path}")
    
    with open(base_config_path, 'r') as f:
        config = json.load(f)
    
    # 폴더명에서 모달리티 자동 추론
    modality_list = parse_modality_from_folder(folder_name)
    
    print(f"📝 설정을 eval 모드로 수정...")
    print(f"  🔍 폴더명 분석: {folder_name}")
    print(f"  📊 추론된 모달리티: {modality_list}")
    
    # eval 모드로 수정
    original_mode = config.get('mode', 'train')
    original_modality = config.get('modality', ['RGB', 'Gyro', 'Acce'])
    
    config['mode'] = 'eval'
    config['enable_ood'] = False
    config['weights_path'] = weights_path
    config['use_wandb'] = False
    config['modality'] = modality_list
    
    print(f"  🔄 mode: {original_mode} → eval")
    print(f"  🔄 modality: {original_modality} → {modality_list}")
    print(f"  🔄 enable_ood: → False")
    print(f"  🔄 weights_path: → {weights_path}")
    print(f"  🔄 use_wandb: → False")
    
    return config

# ================= 모델 평가 =================
def evaluate_upperbound_model(config):
    """Upperbound 모델 평가 → y_true, y_pred 반환"""
    print(f"\n🚀 Upperbound 모델 평가 시작")
    print(f"  📊 모달리티: {config['modality']}")
    print(f"  🎯 클래스 수: {config['init_cls']}")
    
    try:
        # 랜덤 시드 및 디바이스 설정
        set_random_seed(config["seed"])
        config["device"] = set_device(config["device"])
        
        print(f"  🔧 디바이스: {config['device']}")
        
        # 모델 생성
        model = get_model(config["model_name"], config)
        
        # 데이터 매니저 생성
        image_tmpl = {}
        for m in config["modality"]:
            if m in ['RGB', 'RGBDiff']:
                image_tmpl[m] = "{:06d}.jpg"
            elif m == 'Flow':
                image_tmpl[m] = config.get("flow_prefix", "flow_") + "{}_{:06d}.jpg"
        
        if config["dataset"] == "mmea-tbn":
            data_manager = TBNDataManager(model, image_tmpl, config)
        elif config["dataset"] == "mmea-tsn":
            data_manager = TSNDataManager(model, image_tmpl, config)
        else:
            raise NotImplementedError(f"알 수 없는 데이터셋: {config['dataset']}")
        
        # upperbound 모델 상태 설정
        model._cur_task = 0
        model._total_classes = config['init_cls']
        model._classes_seen_so_far = config['init_cls']
        model.total_classnum = config['init_cls']
        
        print(f"  🎯 모델 상태 설정 완료 (총 {config['init_cls']}개 클래스)")
        
        # 데이터 로더 설정
        test_dataset = data_manager.get_dataset(
            np.arange(0, config['init_cls']), 
            source="test", 
            mode="test"
        )
        from torch.utils.data import DataLoader
        model.test_loader = DataLoader(
            test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['workers']
        )
        print(f"  📚 테스트 데이터: {len(model.test_loader.dataset)}개 샘플")
        
        # 체크포인트 찾기 및 로드
        weights_path = config['weights_path']
        checkpoint_files = [f for f in os.listdir(weights_path) if f.endswith('.pkl')]
        if not checkpoint_files:
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {weights_path}")
        
        checkpoint_path = os.path.join(weights_path, checkpoint_files[0])
        print(f"  📥 체크포인트 로드: {checkpoint_path}")
        
        # classifier 크기를 먼저 업데이트한 후 체크포인트 로드
        model._update_classifier(config['init_cls'])
        print(f"  🔧 Classifier 업데이트 완료: {config['init_cls']}개 클래스")
        
        # 체크포인트 로드
        model.load_checkpoint(checkpoint_path)
        model._network = model._network.to(model._device)
        model._network.eval()
        
        # 디버그: classifier 상태 확인
        if hasattr(model._network, 'fc') and model._network.fc is not None:
            print(f"  ✅ Classifier 정상 로드됨: {type(model._network.fc)}")
        else:
            print(f"  ❌ Classifier가 None입니다!")
            raise RuntimeError("Classifier가 제대로 로드되지 않았습니다")
        
        # _eval_cnn을 직접 호출하여 y_true, y_pred 얻기
        print(f"  🔄 evaluate_cl() 호출하여 예측 결과 추출...")
        y_pred, y_true = model._eval_cnn(model.test_loader)
        
        # topk에서 첫 번째 예측만 사용
        y_pred_single = y_pred[:, 0]
        
        # 정확도 계산
        accuracy = (y_pred_single == y_true).mean()
        
        print(f"  ✅ 예측 완료!")
        print(f"  📊 총 샘플 수: {len(y_true)}")
        print(f"  🎯 정확도: {accuracy:.3f}")
        
        # evaluate_cl()도 호출해서 기존 메트릭도 확인
        print(f"  📊 evaluate_cl() 호출하여 기존 메트릭 확인...")
        cl_results = model.evaluate_cl()
        print(f"  📈 CL 결과 - CNN: {cl_results['cnn']['top1']:.2f}%")
        
        return y_true, y_pred_single, accuracy, cl_results, True
        
    except Exception as e:
        print(f"  ❌ 모델 평가 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, False

# ================= 시각화 함수 =================
def create_confusion_matrix_plot(y_true, y_pred, class_names, title, save_path):
    """검증된 고품질 confusion matrix 시각화"""
    
    print(f"🎨 검증된 Confusion matrix 생성 중...")
    
    # 🔍 클래스 매핑 검증 정보
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    print(f"  📊 데이터 검증:")
    print(f"    - y_true 클래스 범위: {unique_true.min()} ~ {unique_true.max()} ({len(unique_true)}개)")
    print(f"    - y_pred 클래스 범위: {unique_pred.min()} ~ {unique_pred.max()} ({len(unique_pred)}개)")
    
    # 🎯 첫 번째 클래스 (클래스 26 = open_close_door) 검증
    first_class_samples = np.sum(y_true == 0)  # 학습순서 0번
    first_class_correct = np.sum((y_true == 0) & (y_pred == 0))
    print(f"    - 첫 번째 클래스 (원본 ID 26, 'open_close_door'):")
    print(f"      실제 샘플: {first_class_samples}개, 정확 예측: {first_class_correct}개")
    
    # Confusion matrix 계산
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    # 정규화 (각 행의 합이 1이 되도록)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    # 추가 메트릭 계산
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # 클래스별 정확도 계산
    class_accuracies = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    
    print(f"  📊 메트릭:")
    print(f"    - Macro Precision: {macro_precision:.3f}")
    print(f"    - Macro Recall: {macro_recall:.3f}")
    print(f"    - Macro F1-Score: {macro_f1:.3f}")
    print(f"    - 최고 클래스 정확도: {np.max(class_accuracies):.3f}")
    print(f"    - 최저 클래스 정확도: {np.min(class_accuracies):.3f}")
    print(f"    - 첫 번째 클래스 정확도: {class_accuracies[0]:.3f}")
    
    # 그래프 설정 - 고해상도
    plt.figure(figsize=(28, 24))
    
    # 커스텀 컬러맵: 0은 검은색, 1은 흰색, 중간은 빨간색
    colors = ['#000000', '#FF0000', '#FFFFFF']  # 검은색 -> 빨간색 -> 흰색
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
    
    # Heatmap 생성
    ax = sns.heatmap(cm_normalized, 
                     annot=True, 
                     fmt='.2f',
                     cmap=cmap,
                     xticklabels=class_names,
                     yticklabels=class_names,
                     cbar_kws={'label': 'Normalized Probability', 'shrink': 0.8},
                     square=True,
                     annot_kws={'size': 7, 'weight': 'bold'},
                     linewidths=0.5,
                     linecolor='gray')
    
    # 제목 및 라벨
    plt.title(title, fontsize=24, fontweight='bold', pad=30)
    plt.xlabel('Predicted Label', fontsize=18, fontweight='bold')
    plt.ylabel('True Label', fontsize=18, fontweight='bold')
    
    # 축 레이블 설정 - X축 레이블을 세로로 회전
    plt.xticks(rotation=90, ha='center', fontsize=10, fontweight='bold')
    plt.yticks(rotation=0, fontsize=10, fontweight='bold')
    
    # 🔥 대각선 강조 (정확한 예측) - 파란색 박스만
    for i in range(len(class_names)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=4))
    
    # 레이아웃 조정 및 저장
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', transparent=False)
    plt.close()
    
    print(f"  ✅ 검증된 Confusion matrix 저장: {save_path}")
    
    # 메트릭 반환
    metrics = {
        'confusion_matrix': cm,
        'normalized_cm': cm_normalized,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'class_accuracies': class_accuracies
    }
    
    return cm, cm_normalized, metrics

def create_difference_confusion_matrix(cm1, cm2, modality1, modality2, class_names, title, save_path):
    """검증된 차이 confusion matrix 시각화 (대각선 강조 포함)"""
    
    print(f"🎨 검증된 차이 confusion matrix 생성 중...")
    print(f"  📊 {modality1} vs {modality2}")
    
    # 차이 계산 (cm1 - cm2)
    diff_cm = cm1 - cm2
    
    # 🎯 첫 번째 클래스 차이 검증
    first_class_diff = diff_cm[0, 0]
    print(f"  🎯 첫 번째 클래스 (원본 ID 26, 'open_close_door') 차이: {first_class_diff:.3f}")
    
    # 그래프 설정 - 고해상도
    plt.figure(figsize=(28, 24))
    
    # 차이 시각화용 컬러맵: 음수는 빨간색, 0은 흰색, 양수는 파란색
    colors = ['#FF0000', '#FFFFFF', '#0000FF']  # 빨간색 -> 흰색 -> 파란색
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('diff', colors, N=n_bins)
    
    # 컬러 스케일 범위 설정 (대칭으로)
    vmax = max(abs(diff_cm.min()), abs(diff_cm.max()))
    if vmax == 0:
        vmax = 0.1  # 모든 값이 0인 경우 방지
    
    # Heatmap 생성
    ax = sns.heatmap(diff_cm * 100,  # 퍼센트로 변환
                     annot=True, 
                     fmt='.1f',
                     cmap=cmap,
                     center=0,  # 중앙값을 0으로 설정
                     vmin=-vmax*100, 
                     vmax=vmax*100,
                     xticklabels=class_names,
                     yticklabels=class_names,
                     cbar_kws={'label': 'Difference (%)', 'shrink': 0.8},
                     square=True,
                     annot_kws={'size': 6, 'weight': 'bold'},
                     linewidths=0.3,
                     linecolor='gray')
    
    # 제목 및 라벨
    plt.title(title, fontsize=24, fontweight='bold', pad=30)
    plt.xlabel('Predicted Label', fontsize=18, fontweight='bold')
    plt.ylabel('True Label', fontsize=18, fontweight='bold')
    
    # 축 레이블 설정 - X축 레이블을 세로로 회전
    plt.xticks(rotation=90, ha='center', fontsize=10, fontweight='bold')
    plt.yticks(rotation=0, fontsize=10, fontweight='bold')
    
    # 🔥 대각선 강조 - 검은색 박스만 단순하게
    for i in range(len(class_names)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=4))
    
    # 레이아웃 조정 및 저장
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', transparent=False)
    plt.close()
    
    print(f"  ✅ 검증된 차이 confusion matrix 저장: {save_path}")
    
    # 통계 정보 계산
    stats = {
        'max_improvement': float(diff_cm.max()),
        'max_degradation': float(diff_cm.min()),
        'mean_difference': float(diff_cm.mean()),
        'total_positive_changes': int((diff_cm > 0).sum()),
        'total_negative_changes': int((diff_cm < 0).sum()),
        'diagonal_improvement': float(diff_cm.diagonal().mean()),
        'first_class_diff': float(first_class_diff)  # 첫 번째 클래스 차이 추가
    }
    
    return diff_cm, stats

# ================= 모델 탐색 =================
def find_upperbound_models():
    """upperbound 모델들을 자동으로 찾기"""
    weights_base_dir = "/workspace/MMEA-OWCL/weights"
    
    print("🔍 Upperbound 모델 검색 중...")
    print(f"  📁 검색 디렉토리: {weights_base_dir}")
    
    # upperbound가 포함된 모든 폴더 찾기
    upperbound_folders = []
    if os.path.exists(weights_base_dir):
        for folder in os.listdir(weights_base_dir):
            if "upperbound" in folder and os.path.isdir(os.path.join(weights_base_dir, folder)):
                upperbound_folders.append(os.path.join(weights_base_dir, folder))
    
    print(f"  📊 발견된 upperbound 모델: {len(upperbound_folders)}개")
    
    # 각 폴더에서 가장 최신 타임스탬프 찾기
    model_paths = []
    for folder in upperbound_folders:
        folder_name = os.path.basename(folder)
        print(f"    🔍 검사 중: {folder_name}")
        
        # 타임스탬프 디렉토리 찾기
        timestamp_dirs = []
        if os.path.exists(folder):
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                if os.path.isdir(item_path) and item.startswith("Sep"):
                    timestamp_dirs.append(item)
        
        if timestamp_dirs:
            # 가장 최신 타임스탬프 선택
            latest_timestamp = max(timestamp_dirs)
            weights_path = os.path.join(folder, latest_timestamp)
            
            # 체크포인트 파일 존재 확인
            checkpoint_path = os.path.join(weights_path, "task_0_checkpoint_0.pkl")
            if os.path.exists(checkpoint_path):
                model_paths.append({
                    'folder_name': folder_name,
                    'weights_path': weights_path,
                    'checkpoint_path': checkpoint_path,
                    'timestamp': latest_timestamp
                })
                print(f"      ✅ 유효한 모델: {latest_timestamp}")
            else:
                print(f"      ❌ 체크포인트 없음: {checkpoint_path}")
        else:
            print(f"      ❌ 타임스탬프 디렉토리 없음")
    
    print(f"\n📋 최종 발견된 유효한 모델: {len(model_paths)}개")
    for model in model_paths:
        print(f"  - {model['folder_name']} ({model['timestamp']})")
    
    return model_paths

# ================= 메인 =================
def main():
    """메인 함수 - 여러 모달리티에 대한 confusion matrix 생성 + 차이 분석"""
    print("🎯 Multi-Modality Upperbound Confusion Matrix + Difference Analysis")
    print("=" * 80)
    
    # 기본 설정
    base_config_path = "/workspace/MMEA-OWCL/exps/exp_tbn_upperbound.json"
    
    # 파일 존재 확인
    if not os.path.exists(base_config_path):
        print(f"❌ 설정 파일을 찾을 수 없습니다: {base_config_path}")
        return
    
    # upperbound 모델들 자동 검색
    model_paths = find_upperbound_models()
    
    if not model_paths:
        print("❌ 유효한 upperbound 모델을 찾을 수 없습니다")
        return
    
    # 학습 순서에 따른 클래스 이름들
    ordered_class_names = [ORIGINAL_CLASS_NAMES[COMMON_CLASS_ORDER[i]] for i in range(len(COMMON_CLASS_ORDER))]
    
    # 통합 출력 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/workspace/MMEA-OWCL/upperbound_confusion_matrices_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 통합 결과 저장 디렉토리: {output_dir}")
    
    # 모든 모델에 대한 결과 저장
    all_results = {}
    all_confusion_matrices = {}  # 🔥 confusion matrix 저장용
    successful_models = 0
    failed_models = 0
    
    # ================= STEP 1: 각 모델에 대해 confusion matrix 생성 =================
    print(f"\n{'='*80}")
    print("🚀 STEP 1: 개별 모달리티 Confusion Matrix 생성")
    print(f"{'='*80}")
    
    for i, model_info in enumerate(model_paths, 1):
        folder_name = model_info['folder_name']
        weights_path = model_info['weights_path']
        
        print(f"\n{'='*50}")
        print(f"🚀 [{i}/{len(model_paths)}] 처리 중: {folder_name}")
        print(f"{'='*50}")
        
        try:
            # 1. 설정 로드 및 수정 (폴더명 기반 모달리티 자동 추론)
            config = load_and_modify_config(base_config_path, weights_path, folder_name)
            
            # 2. 모델 평가
            y_true, y_pred, accuracy, cl_results, success = evaluate_upperbound_model(config)
            
            if success and y_true is not None:
                # 3. Confusion matrix 생성
                modality_name = '+'.join(config['modality'])
                safe_modality_name = modality_name.replace('+', '_')
                filename = f"upperbound_{safe_modality_name}_confusion_matrix.png"
                save_path = os.path.join(output_dir, filename)
                title = f"Upperbound - {modality_name}\\nConfusion Matrix (Normalized)\\nAccuracy: {accuracy:.3f}"
                
                cm, cm_norm, metrics = create_confusion_matrix_plot(
                    y_true, y_pred, ordered_class_names, title, save_path
                )
                
                # 4. 결과 저장
                all_results[modality_name] = {
                    'folder_name': folder_name,
                    'weights_path': weights_path,
                    'modality': config['modality'],
                    'accuracy': float(accuracy),
                    'num_samples': len(y_true),
                    'confusion_matrix_path': save_path,
                    'cl_results': cl_results,
                    'macro_precision': float(metrics['macro_precision']),
                    'macro_recall': float(metrics['macro_recall']),
                    'macro_f1': float(metrics['macro_f1']),
                    'best_class_accuracy': float(np.max(metrics['class_accuracies'])),
                    'worst_class_accuracy': float(np.min(metrics['class_accuracies'])),
                    'timestamp': model_info['timestamp'],
                    # 클래스별 세부 메트릭 추가
                    'per_class_metrics': {}
                }
                
                # 클래스별 세부 메트릭 저장
                for j, class_name in enumerate(ordered_class_names):
                    all_results[modality_name]['per_class_metrics'][class_name] = {
                        'precision': float(metrics['precision'][j]),
                        'recall': float(metrics['recall'][j]),
                        'f1_score': float(metrics['f1_score'][j]),
                        'support': int(metrics['support'][j]),
                        'accuracy': float(metrics['class_accuracies'][j])
                    }
                
                # 🔥 중요: confusion matrix 저장 (차이 분석용)
                all_confusion_matrices[modality_name] = cm_norm
                
                successful_models += 1
                print(f"  ✅ {modality_name}: 성공 (정확도: {accuracy:.3f})")
                
            else:
                failed_models += 1
                print(f"  ❌ {folder_name}: 평가 실패")
                
        except Exception as e:
            failed_models += 1
            print(f"  ❌ {folder_name}: 오류 발생 - {e}")
    
    # ================= STEP 2: 차이 분석 Confusion Matrix 생성 =================
    print(f"\n{'='*80}")
    print("🚀 STEP 2: 모달리티 간 차이 분석 Confusion Matrix 생성")
    print(f"{'='*80}")
    
    # 비교할 모달리티 조합들 정의
    comparisons = [
        ('RGB+Gyro+Acce', 'RGB', 'All vs RGB'),
        ('RGB+Gyro', 'RGB', 'RGB+Gyro vs RGB'),  
        ('RGB+Acce', 'RGB', 'RGB+Acce vs RGB'),
        ('RGB+Gyro+Acce', 'RGB+Gyro', 'All vs RGB+Gyro'),
        ('RGB+Gyro+Acce', 'RGB+Acce', 'All vs RGB+Acce'),
        ('RGB+Gyro', 'RGB+Acce', 'RGB+Gyro vs RGB+Acce'),
        ('Gyro+Acce', 'Gyro', 'Gyro+Acce vs Gyro'),
        ('Gyro+Acce', 'Acce', 'Gyro+Acce vs Acce'),
        ('RGB+Gyro+Acce', 'Gyro+Acce', 'All vs Gyro+Acce')
    ]
    
    diff_results = {}
    
    print(f"\n🔍 차이 분석 시작...")
    print(f"  📊 총 {len(comparisons)}개 비교 조합")
    
    for comparison_idx, (modality1, modality2, comparison_name) in enumerate(comparisons, 1):
        print(f"\n{'='*50}")
        print(f"🚀 [{comparison_idx}/{len(comparisons)}] 비교: {comparison_name}")
        print(f"{'='*50}")
        
        # Confusion matrix 확인
        if modality1 not in all_confusion_matrices:
            print(f"  ❌ {modality1} confusion matrix를 찾을 수 없습니다")
            continue
        if modality2 not in all_confusion_matrices:
            print(f"  ❌ {modality2} confusion matrix를 찾을 수 없습니다")
            continue
        
        cm1 = all_confusion_matrices[modality1]
        cm2 = all_confusion_matrices[modality2]
        
        # 파일명 생성
        safe_name = comparison_name.replace(' ', '_').replace('+', '_').replace('vs', 'vs')
        filename = f"diff_{safe_name}_confusion_matrix.png"
        save_path = os.path.join(output_dir, filename)
        
        # 제목 생성
        acc1 = all_results[modality1]['accuracy']
        acc2 = all_results[modality2]['accuracy']
        title = f"Difference: {comparison_name}\\nConfusion Matrix Difference\\n{modality1}({acc1:.3f}) - {modality2}({acc2:.3f})"
        
        # 차이 confusion matrix 생성
        diff_cm, stats = create_difference_confusion_matrix(
            cm1, cm2, modality1, modality2, ordered_class_names, title, save_path
        )
        
        # 결과 저장
        diff_results[comparison_name] = {
            'modality1': modality1,
            'modality2': modality2,
            'accuracy1': float(acc1),
            'accuracy2': float(acc2),
            'accuracy_diff': float(acc1 - acc2),
            'save_path': save_path,
            'stats': stats
        }
        
        print(f"  📊 정확도 차이: {acc1:.3f} - {acc2:.3f} = {acc1-acc2:.3f}")
        print(f"  📈 대각선 평균 개선: {stats['diagonal_improvement']:.3f}")
        print(f"  🎯 첫 번째 클래스 차이: {stats['first_class_diff']:.3f}")
        print(f"  ✅ 저장 완료: {filename}")
    
    # ================= STEP 3: 최종 결과 요약 =================
    print(f"\n{'='*80}")
    print("📊 최종 결과 요약")
    print(f"{'='*80}")
    print(f"  🎯 총 처리된 모델: {len(model_paths)}개")
    print(f"  ✅ 성공한 모델: {successful_models}개")
    print(f"  ❌ 실패한 모델: {failed_models}개")
    print(f"  🔍 생성된 차이 분석: {len(diff_results)}개")
    print(f"  📁 결과 저장 위치: {output_dir}")
    
    if all_results:
        print(f"\n🎨 생성된 Confusion Matrix:")
        print("-" * 70)
        print(f"{'Modality':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 70)
        
        # 정확도 순으로 정렬
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for modality, result in sorted_results:
            print(f"{modality:<20} {result['accuracy']:<10.3f} {result['macro_precision']:<10.3f} "
                  f"{result['macro_recall']:<10.3f} {result['macro_f1']:<10.3f}")
        
        # 성능 분석
        if len(sorted_results) > 1:
            best_modality = sorted_results[0]
            worst_modality = sorted_results[-1]
            print(f"\n📈 성능 분석:")
            print(f"  🥇 최고 성능: {best_modality[0]} ({best_modality[1]['accuracy']:.3f})")
            print(f"  📉 최저 성능: {worst_modality[0]} ({worst_modality[1]['accuracy']:.3f})")
            print(f"  📊 성능 차이: {best_modality[1]['accuracy'] - worst_modality[1]['accuracy']:.3f}")
            
            # 평균 성능
            avg_accuracy = np.mean([r['accuracy'] for r in all_results.values()])
            print(f"  📊 평균 정확도: {avg_accuracy:.3f}")
        
        # 차이 분석 결과
        if diff_results:
            print(f"\n📈 차이 분석 결과:")
            print("-" * 70)
            print(f"{'Comparison':<30} {'Acc Diff':<10} {'Diag Improve':<12}")
            print("-" * 70)
            
            # 정확도 차이 순으로 정렬
            sorted_diffs = sorted(diff_results.items(), key=lambda x: x[1]['accuracy_diff'], reverse=True)
            
            for comparison, result in sorted_diffs:
                acc_diff = result['accuracy_diff']
                diag_improve = result['stats']['diagonal_improvement']
                print(f"{comparison:<30} {acc_diff:<10.3f} {diag_improve:<12.3f}")
        
        # 통합 JSON 저장
        summary_data = {
            'individual_results': all_results,
            'difference_analysis': diff_results,
            'summary': {
                'total_models': len(model_paths),
                'successful_models': successful_models,
                'failed_models': failed_models,
                'total_comparisons': len(diff_results),
                'best_modality': sorted_results[0][0] if sorted_results else None,
                'best_accuracy': sorted_results[0][1]['accuracy'] if sorted_results else None,
                'average_accuracy': float(np.mean([r['accuracy'] for r in all_results.values()])) if all_results else None,
                'generation_time': timestamp
            }
        }
        
        summary_path = os.path.join(output_dir, "all_upperbound_results.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n📋 통합 결과 요약: {summary_path}")
        
        print(f"\n✅ 모든 Upperbound Confusion Matrix 및 차이 분석 완료!")
        print(f"📁 결과 확인: {output_dir}")
        
    else:
        print(f"\n❌ 성공한 모델이 없습니다")

if __name__ == "__main__":
    main()