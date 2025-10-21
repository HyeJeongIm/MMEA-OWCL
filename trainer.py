import os
import sys
import logging
import datetime

from models.model_factory import get_model
from utils.utils import set_random_seed, set_device
from dataloader.data_manager import TBNDataManager, TSNDataManager


def train(args):
    """메인 훈련 함수"""
    
    # 실험 디렉토리 생성
    modality_str = ''.join(args["modality"]).lower()
    suffix = args.get("experiment_suffix", "")
    
    # 모달리티 정보 처리
    modality_list = args.get('modality', ['RGB'])
    modality_str = ''.join(modality_list).lower()  # ['RGB', 'Gyro', 'Acce'] -> 'rgbgyroacce'
    
    # partialbn과 freeze 정보
    pb_flag = "1" if args.get('partialbn', False) else "0"
    fr_flag = "1" if args.get('freeze', False) else "0"
    
    # fusion_type 정보 추가 (기본값: concat)
    fusion_type = args.get('fusion_type', 'concat')
    
    experiment_name_parts = [
        args['dataset'],
        args['model_name'],
        fusion_type,
        modality_str,
        f"ep{args['epochs']}",
        f"bs{args['batch_size']}",
        f"pb{pb_flag}",
        f"fr{fr_flag}",
        f"inc{args['increment']}",
        f"mem{args['memory_size']}"
    ]
    
    # auxiliary_head를 사용하는 경우에만 confidence_method를 경로에 추가
    if 'auxiliary_head' in fusion_type:
        confidence_method = args.get('confidence_method', 'energy')
        experiment_name_parts.append(confidence_method)
    
    if suffix:
        experiment_name_parts.append(suffix)
    
    experiment_name = '_'.join(experiment_name_parts)
    
    experiment_dir = os.path.join(experiment_name, f"seed_{args['seed']}")
    log_dir = os.path.join("logs", experiment_dir)
    weights_dir = os.path.join(log_dir, "weights")
    results_dir = os.path.join(log_dir, "results")
    
    # mode에 따른 디렉토리 생성
    mode = args.get('mode', 'train')
    
    if mode == 'eval':
        # eval 모드: results/ 폴더 안에 실험 디렉토리 생성
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"✓ [EVAL] 실험 디렉토리 생성: {experiment_dir}")
        print(f"✓ [EVAL] 결과 저장 경로: {results_dir}")
        print(f"✓ [EVAL] 모델 가중치 로드 경로: {weights_dir}")
        
        # train 모드에서는 파일 로깅도 포함
        log_name = f"eval_{args['prefix']}_{args['seed']}_{args['model_name']}_" \
                   f"{args['dataset']}_{args['init_cls']}_{args['increment']}.log"
        log_path = os.path.join(results_dir, log_name)
        
        # eval 모드에서는 간단한 로거 설정 (콘솔 출력만)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s] => %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout),
            ]
        )
    else:
        # train 모드: 기존과 동일 (weights/, logs/ 폴더)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(weights_dir, exist_ok=True)
        
        print(f"✓ [TRAIN] 실험 디렉토리 생성: {experiment_dir}")
        print(f"✓ [TRAIN] 로그 저장 경로: {log_dir}")
        print(f"✓ [TRAIN] 모델 가중치 저장 경로: {weights_dir}")
        
        # train 모드에서는 파일 로깅도 포함
        log_name = f"{args['prefix']}_{args['seed']}_{args['model_name']}_" \
                   f"{args['dataset']}_{args['init_cls']}_{args['increment']}.log"
        log_path = os.path.join(log_dir, log_name)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s] => %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout),
            ]
        )
        
        # 설정 저장
        with open(os.path.join(log_dir, "args.txt"), "w") as f:
            f.write(str(args))
    
    # 랜덤 시드 및 디바이스 설정
    set_random_seed(args["seed"])
    args["device"] = set_device(args["device"])
    
    # mode에 따른 실행
    if mode == 'eval':
        # eval 모드에서는 weights_dir, log_dir가 없으므로 None 전달
        _train(args, experiment_dir, weights_dir, None)
    else:
        # train 모드
        _train(args, experiment_dir, weights_dir, log_dir)


def _train(args, experiment_dir, weights_dir, log_dir):
    """핵심 훈련 루프"""

    model = get_model(args["model_name"], args)
    
    # 모델 설정
    if args["partialbn"]:
        model._network.backbone.freeze_fn('partialbn_parameters') # RGB modality의 첫 번쨰 이후 BatchNorm2D 파라미터 고정
    if args["freeze"]:
        model._network.backbone.freeze()
    
    # 이미지 템플릿 설정
    image_tmpl = {}
    for m in args["modality"]:
        if m in ['RGB', 'RGBDiff']:
            image_tmpl[m] = "{:06d}.jpg"
        elif m == 'Flow':
            image_tmpl[m] = args["flow_prefix"] + "{}_{:06d}.jpg"
    
    # 데이터 매니저 초기화
    if "tbn" in args["model_name"]:
        data_manager = TBNDataManager(model, image_tmpl, args)
    elif "tsn" in args["model_name"]:
        data_manager = TSNDataManager(model, image_tmpl, args)
    else:
        raise NotImplementedError(f"알 수 없는 데이터셋: {args['dataset']}")
    
    # 결과 저장소 초기화
    all_cl_results = {}
    all_ood_results = None
    
    # 실행 모드 결정
    execution_mode = args.get("mode", "train")
    
    if execution_mode == "eval":
        all_ood_results = _run_eval_mode(args, model, data_manager, weights_dir, all_cl_results)
    elif execution_mode == "train":
        _run_training_mode(args, model, data_manager, weights_dir, all_cl_results)
    elif execution_mode == "upperbound":
        _run_upperbound_mode(args, model, data_manager, weights_dir, all_cl_results)
        # Upper-bound는 단일 태스크이므로 nb_tasks=1로 설정
        _log_final_summary(all_cl_results, 1, all_ood_results)
    else:
        raise ValueError(f"Unknown mode: {execution_mode}. Use 'train', 'eval', or 'upperbound'")
    
    # 일반 모드의 최종 결과 요약
    if execution_mode != "upperbound":
        _log_final_summary(all_cl_results, data_manager.nb_tasks, all_ood_results)
    


def _check_inference_mode(args, weights_dir):
    """Check if we should run in inference mode (pre-trained weights exist)"""
    if not args.get("enable_ood", False):
        return False
    
    # Check if any checkpoint files exist in weights directory
    if os.path.exists(weights_dir):
        checkpoint_files = [f for f in os.listdir(weights_dir) if f.endswith('.pkl')]
        if checkpoint_files:
            logging.info(f"✅ Found {len(checkpoint_files)} checkpoint(s) in {weights_dir}")
            logging.info("🔄 Running in INFERENCE MODE (using pre-trained weights)")
            return True
    
    logging.info("🚀 Running in TRAINING MODE (training from scratch)")
    return False


# CL 모델 학습 
def _run_training_mode(args, model, data_manager, weights_dir, all_cl_results):
    """Run training mode: train each task and evaluate"""
    logging.info("=== TRAINING MODE ===")
    
    # OOD 결과 저장소 초기화
    all_ood_results = {}
    
    for task_id in range(data_manager.nb_tasks):
        print(f"\nTask {task_id + 1}/{data_manager.nb_tasks} 시작")
        
        # 증분 학습
        model.incremental_train(data_manager)
        
        # CL 평가 (tuple unpacking)
        cl_results, cl_metrics = model.evaluate_cl()
        
        # OOD 평가 (enable_ood=True인 경우에만)
        if args.get("enable_ood", False):
            ood_results, score_distributions, ood_metrics = model.evaluate_ood()
            # OOD 결과 저장
            task_key = f"task_{task_id}"
            all_ood_results[task_key] = ood_results
            # Wandb 로깅 (CL + OOD)
            model.auto_wandb_log(cl_metrics, ood_metrics, task_id + 1)
        else:
            # Wandb 로깅 (CL only)
            model.auto_wandb_log(cl_metrics, {}, task_id + 1)
        
        # 결과 저장
        task_key = f"task_{task_id}"
        all_cl_results[task_key] = cl_results
        
        # 메모리 정리
        if hasattr(model, 'clear_cached_data'):
            model.clear_cached_data()
        
        # 모델 상태 업데이트
        model.after_task()
        
        # 체크포인트 저장
        try:
            model.save_checkpoint(weights_dir, f"task_{task_id}_checkpoint")
        except AttributeError:
            pass
        
        # 태스크 요약
        cl_acc = cl_results['cnn']['top1']
        print(f"Task {task_id + 1} 완료 - CL 정확도: {cl_acc:.1f}%")
    
    # 🎯 모든 task 완료 후 평균 메트릭 계산 및 로깅
    if args.get("enable_ood", False) and all_ood_results and args.get("use_wandb", False):
        _log_average_ood_metrics(all_ood_results, args)


def _run_eval_mode(args, model, data_manager, weights_dir, all_cl_results):
    """Run evaluation mode: load pre-trained weights and evaluate OOD only"""
    logging.info("=== EVALUATION MODE ===")
    
    # 가중치 경로 확인    
    if not os.path.exists(weights_dir):
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
    
    # OOD 결과 저장소 초기화
    all_ood_results = {}
    
    for task_id in range(data_manager.nb_tasks):
        print(f"\nTask {task_id + 1}/{data_manager.nb_tasks} OOD 평가 시작")
        
        # 체크포인트 경로 찾기
        checkpoint_path = os.path.join(weights_dir, f"task_{task_id}_checkpoint_{task_id}.pkl")
        
        try:
            # Evaluation 모드로 OOD 평가 실행
            if args.get("enable_ood", False) and task_id != data_manager.nb_tasks - 1:
                cl_results, ood_results, score_distributions = model.inference_mode_evaluation(
                    data_manager, checkpoint_path, task_id
                )
                logging.info("✅ OOD 평가 완료")
                
                # OOD 결과 저장
                task_key = f"task_{task_id}"
                all_ood_results[task_key] = ood_results
            else:
                logging.warning("enable_ood=False in eval mode. Only CL evaluation will be performed.")
                model.enable_ood = False if model.enable_ood == True else model.enable_ood
                cl_results = model.inference_mode_evaluation(
                    data_manager, checkpoint_path, task_id
                )
            
            # CL 결과 저장
            task_key = f"task_{task_id}"
            all_cl_results[task_key] = cl_results
            
            # 메모리 정리
            if hasattr(model, 'clear_cached_data'):
                model.clear_cached_data()
            
            # 태스크 요약
            cl_acc = cl_results['cnn']['top1']
            print(f"Task {task_id + 1} 평가 완료 - CL 정확도: {cl_acc:.1f}%")
            
        except Exception as e:
            logging.error(f"Task {task_id + 1} evaluation failed: {e}")
            continue
    
    # 🎯 모든 task 완료 후 평균 메트릭 계산 및 로깅
    if args.get("enable_ood", False) and all_ood_results and args.get("use_wandb", False):
        _log_average_ood_metrics(all_ood_results, args)
    
    # OOD 결과가 있으면 반환
    return all_ood_results if all_ood_results else None


def _log_average_ood_metrics(all_ood_results, args):
    """
    모든 task의 OOD 메트릭 평균 계산 및 Wandb 로깅
    
    Args:
        all_ood_results: {
            'task_0': {'Energy': {'auroc': 85.3, 'aupr_id': 88.7, ...}},
            'task_1': {'Energy': {'auroc': 87.1, 'aupr_id': 90.2, ...}},
            ...
        }
        args: 설정 딕셔너리
    """
    import wandb
    
    logging.info("\n" + "="*70)
    logging.info("📊 Computing Average OOD Metrics Across All Tasks")
    logging.info("="*70)
    
    # Method별 메트릭 수집
    method_metrics = {}  # {'Energy': {'auroc': [85.3, 87.1, ...], 'aupr_id': [88.7, 90.2, ...], ...}}
    
    for task_key, ood_results in all_ood_results.items():
        for method_name, metrics in ood_results.items():
            if 'error' in metrics:
                continue  # 에러가 있는 결과는 제외
            
            if method_name not in method_metrics:
                method_metrics[method_name] = {}
            
            # 각 메트릭 수집
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):  # 숫자형 메트릭만
                    if metric_name not in method_metrics[method_name]:
                        method_metrics[method_name][metric_name] = []
                    method_metrics[method_name][metric_name].append(metric_value)
    
    # 평균 계산 및 로깅
    avg_metrics = {}
    
    for method_name, metrics in method_metrics.items():
        logging.info(f"\n🔍 {method_name} - Average Metrics:")
        
        for metric_name, values in metrics.items():
            if len(values) > 0:
                avg_value = sum(values) / len(values)
                
                # Wandb 키 생성
                wandb_key = f"Average_OOD/{method_name}_{metric_name}"
                avg_metrics[wandb_key] = avg_value
                
                # 콘솔 로그
                logging.info(f"   {metric_name}: {avg_value:.2f}% (over {len(values)} tasks)")
    
    # Wandb에 평균 메트릭 로깅
    if avg_metrics:
        wandb.log(avg_metrics)
        logging.info(f"\n✅ Logged {len(avg_metrics)} average OOD metrics to wandb")
        logging.info("="*70)


def _log_final_summary(cl_results, nb_tasks, ood_results=None):
    """Log comprehensive final results summary"""
    logging.info(f"\n{'='*60}")
    logging.info("FINAL RESULTS SUMMARY")
    logging.info(f"{'='*60}")
    
    if not cl_results:
        logging.warning("No results to summarize")
        return
    
    # Upper-bound vs Regular CL Performance Summary
    if "upperbound" in cl_results:
        # Upper-bound mode
        logging.info("UPPER-BOUND PERFORMANCE:")
        final_cnn_results = cl_results["upperbound"]['cnn']
        final_nme_results = cl_results["upperbound"]['nme']
        
        logging.info(f"  Upper-bound Performance on ALL 32 classes:")
    else:
        # Regular CL mode
        logging.info("CONTINUAL LEARNING PERFORMANCE:")
        task_key = f"task_{nb_tasks - 1}"
        if task_key not in cl_results:
            logging.warning(f"Final task {task_key} results not found")
            return
            
        final_cnn_results = cl_results[task_key]['cnn']
        final_nme_results = cl_results[task_key]['nme']
    
    msg_acc = f"  [Final Avg] FC Acc: {final_cnn_results['top1']:.2f}%"
    msg_grouped_acc = f"  [Final Group] FC Acc: {final_cnn_results['grouped']}"
    
    # CMR_MFN과 같이 메모리가 없는 모델들은 final_nme_results가 None이거나 {}일 수 있음
    if final_nme_results and 'top1' in final_nme_results and final_nme_results['top1'] is not None:
        msg_acc += f", NME Acc: {final_nme_results['top1']:.2f}%"
        msg_grouped_acc += f", NME Acc: {final_nme_results['grouped']}"
    
    logging.info(msg_acc)
    logging.info(msg_grouped_acc)
    
    # OOD Detection Performance Summary
    if ood_results:
        logging.info("\nOOD DETECTION PERFORMANCE:")
        # Get OOD methods from first available task
        first_task_key = None
        for task_id in range(nb_tasks):
            check_key = f"task_{task_id}"
            if check_key in ood_results and ood_results[check_key]:
                first_task_key = check_key
                break
        
        if first_task_key:
            methods = list(ood_results[first_task_key].keys())
            
            # Summary by metric type
            logging.info("  📊 Performance Summary:")
            
            # 전체 평균 계산을 위한 변수들
            overall_auroc_sum = 0
            overall_aupr_id_sum = 0  
            overall_aupr_ood_sum = 0
            overall_methods_count = 0
            all_final_metrics = {}  # 모든 방법론의 최종 평균을 저장
            
            for method in methods:
                avg_auroc = 0
                avg_aupr_id = 0
                avg_aupr_ood = 0
                valid_tasks = 0
                
                for task_id in range(nb_tasks):
                    task_key = f"task_{task_id}"
                    if (task_key in ood_results and method in ood_results[task_key] and 
                        'error' not in ood_results[task_key][method]):
                        avg_auroc += ood_results[task_key][method]['auroc']
                        avg_aupr_id += ood_results[task_key][method]['aupr_id']
                        avg_aupr_ood += ood_results[task_key][method]['aupr_ood']
                        valid_tasks += 1
                
                if valid_tasks > 0:
                    method_avg_auroc = avg_auroc / valid_tasks
                    method_avg_aupr_id = avg_aupr_id / valid_tasks
                    method_avg_aupr_ood = avg_aupr_ood / valid_tasks
                    
                    # 🎯 각 방법론의 최종 단일 점수 계산 (3개 메트릭의 평균)
                    method_final_score = (method_avg_auroc + method_avg_aupr_id + method_avg_aupr_ood) / 3
                    
                    # 개별 방법론 결과 출력
                    logging.info(f"    {method:8}: AUROC={method_avg_auroc:5.1f}% | AUPR_ID={method_avg_aupr_id:5.1f}% | AUPR_OOD={method_avg_aupr_ood:5.1f}% | Final={method_final_score:5.1f}%")
                    
                    # 전체 평균 계산에 추가
                    overall_auroc_sum += method_avg_auroc
                    overall_aupr_id_sum += method_avg_aupr_id
                    overall_aupr_ood_sum += method_avg_aupr_ood
                    overall_methods_count += 1
                    
                    # wandb 로깅용 데이터 저장
                    all_final_metrics[f"Final/{method}_avg_auroc"] = method_avg_auroc
                    all_final_metrics[f"Final/{method}_avg_aupr_id"] = method_avg_aupr_id
                    all_final_metrics[f"Final/{method}_avg_aupr_ood"] = method_avg_aupr_ood
                    all_final_metrics[f"Final/{method}_final_score"] = method_final_score  # 🎯 각 방법론의 최종 단일 점수
                    all_final_metrics[f"Final/{method}_valid_tasks"] = valid_tasks
            
            # 🎯 최종 전체 평균 OOD 성능 계산 및 로깅
            logging.info("\n  🏆 Overall OOD Performance:")
            
            if overall_methods_count > 0:
                final_average_auroc = overall_auroc_sum / overall_methods_count
                final_average_aupr_id = overall_aupr_id_sum / overall_methods_count
                final_average_aupr_ood = overall_aupr_ood_sum / overall_methods_count
                final_average_ood = (final_average_auroc + final_average_aupr_id + final_average_aupr_ood) / 3
                
                logging.info(f"    Average AUROC across all methods: {final_average_auroc:.1f}%")
                logging.info(f"    Average AUPR_ID across all methods: {final_average_aupr_id:.1f}%")
                logging.info(f"    Average AUPR_OOD across all methods: {final_average_aupr_ood:.1f}%")
                logging.info(f"    🏆 Final Average OOD Score: {final_average_ood:.1f}%")
                
                # 전체 평균 메트릭 추가
                all_final_metrics.update({
                    "Final/overall_avg_auroc": final_average_auroc,
                    "Final/overall_avg_aupr_id": final_average_aupr_id, 
                    "Final/overall_avg_aupr_ood": final_average_aupr_ood,
                    "Final/average_ood_score": final_average_ood,
                    "Final/evaluated_methods_count": overall_methods_count
                })
                
                # 🎯 모든 Final 메트릭을 한 번에 wandb에 로깅 (동일한 step)
                try:
                    import wandb
                    if wandb.run is not None and all_final_metrics:
                        logging.info("📊 Logging all final OOD metrics to wandb in a single step...")
                        wandb.log(all_final_metrics)
                        logging.info(f"✅ Logged {len(all_final_metrics)} final metrics to wandb")
                except:
                    pass
            
            # Detailed breakdown by metric
            logging.info("\n  📈 Detailed Breakdown:")
            
            # AUROC comparison
            auroc_scores = []
            for method in methods:
                avg_auroc = 0
                valid_tasks = 0
                for task_id in range(nb_tasks):
                    task_key = f"task_{task_id}"
                    if (task_key in ood_results and method in ood_results[task_key] and 
                        'error' not in ood_results[task_key][method]):
                        avg_auroc += ood_results[task_key][method]['auroc']
                        valid_tasks += 1
                if valid_tasks > 0:
                    auroc_scores.append((method, avg_auroc / valid_tasks))
            
            if auroc_scores:
                auroc_scores.sort(key=lambda x: x[1], reverse=True)
                logging.info("    AUROC Ranking:")
                for i, (method, score) in enumerate(auroc_scores, 1):
                    logging.info(f"      {i}. {method:8}: {score:5.1f}%")
            
            # AUPR_ID comparison  
            aupr_id_scores = []
            for method in methods:
                avg_aupr_id = 0
                valid_tasks = 0
                for task_id in range(nb_tasks):
                    task_key = f"task_{task_id}"
                    if (task_key in ood_results and method in ood_results[task_key] and 
                        'error' not in ood_results[task_key][method]):
                        avg_aupr_id += ood_results[task_key][method]['aupr_id']
                        valid_tasks += 1
                if valid_tasks > 0:
                    aupr_id_scores.append((method, avg_aupr_id / valid_tasks))
            
            if aupr_id_scores:
                aupr_id_scores.sort(key=lambda x: x[1], reverse=True)
                logging.info("    AUPR_ID Ranking (ID Detection):")
                for i, (method, score) in enumerate(aupr_id_scores, 1):
                    logging.info(f"      {i}. {method:8}: {score:5.1f}%")
            
            # AUPR_OOD comparison
            aupr_ood_scores = []
            for method in methods:
                avg_aupr_ood = 0
                valid_tasks = 0
                for task_id in range(nb_tasks):
                    task_key = f"task_{task_id}"
                    if (task_key in ood_results and method in ood_results[task_key] and 
                        'error' not in ood_results[task_key][method]):
                        avg_aupr_ood += ood_results[task_key][method]['aupr_ood']
                        valid_tasks += 1
                if valid_tasks > 0:
                    aupr_ood_scores.append((method, avg_aupr_ood / valid_tasks))
            
            if aupr_ood_scores:
                aupr_ood_scores.sort(key=lambda x: x[1], reverse=True)
                logging.info("    AUPR_OOD Ranking (OOD Detection):")
                for i, (method, score) in enumerate(aupr_ood_scores, 1):
                    logging.info(f"      {i}. {method:8}: {score:5.1f}%")
            
            # FPR95 comparison (lower is better)
            fpr95_scores = []
            for method in methods:
                avg_fpr95 = 0
                valid_tasks = 0
                for task_id in range(nb_tasks):
                    task_key = f"task_{task_id}"
                    if (task_key in ood_results and method in ood_results[task_key] and 
                        'error' not in ood_results[task_key][method] and
                        'fpr95' in ood_results[task_key][method]):
                        avg_fpr95 += ood_results[task_key][method]['fpr95']
                        valid_tasks += 1
                if valid_tasks > 0:
                    fpr95_scores.append((method, avg_fpr95 / valid_tasks))
            
            if fpr95_scores:
                fpr95_scores.sort(key=lambda x: x[1], reverse=False)  # Lower is better
                logging.info("    FPR95 Ranking (Lower is Better):")
                for i, (method, score) in enumerate(fpr95_scores, 1):
                    logging.info(f"      {i}. {method:8}: {score:5.1f}%")
        else:
            logging.info("  No valid OOD results found")
    
    logging.info(f"{'='*60}")


def _run_upperbound_mode(args, model, data_manager, weights_dir, all_cl_results):
    """Run upper-bound mode: train on all classes simultaneously"""
    logging.info("=== UPPER-BOUND MODE ===")
    logging.info("Training on ALL classes simultaneously (no incremental learning)")
    
    # Upper-bound training (single task with all classes)
    logging.info("🎯 Starting Upper-bound Training...")
    model.incremental_train(data_manager)
    
    # Evaluate upper-bound performance
    logging.info("📊 Evaluating Upper-bound Performance...")
    cl_results, cl_metrics = model.evaluate_cl()
    
    # Wandb 로깅
    model.auto_wandb_log(cl_metrics, {}, 0)
    
    # Store results
    all_cl_results["upperbound"] = cl_results
    
    # Save model checkpoint
    try:
        if weights_dir:
            model.save_checkpoint(weights_dir, "upperbound_checkpoint")
            logging.info(f"✅ Upper-bound model saved to: {weights_dir}")
    except AttributeError:
        logging.warning("⚠️ Model does not support save_checkpoint method")
    
    # Update model state
    model.after_task()
    
    # Print summary
    cl_acc = cl_results['cnn']['top1']
    total_classes = data_manager.get_total_classnum()
    logging.info(f"✅ Upper-bound Training Completed!")
    logging.info(f"   📊 Final Accuracy: {cl_acc:.2f}%")
    logging.info(f"   🎯 Total Classes: {total_classes}")
    logging.info(f"   💾 Results saved to: {weights_dir if weights_dir else 'No weights directory'}")
    
    return cl_results