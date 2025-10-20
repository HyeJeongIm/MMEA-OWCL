"""
Multi-Modal Open World Continual Learning (MMOWCL)

Usage:
    # 기본 실행
    python main.py -d mmea -m tbn_replay
    
    # Short options 사용 (빠르고 편리!)
    python main.py -d mmea -m tbn_replay -f auxiliary_head_v2_8 -c energy -g 0
    python main.py -d mmea -m tbn_replay -f auxiliary_head_v2_7 -c max_prob -g 1
    
    # Long options 사용 (명확함)
    python main.py -d mmea -m tbn_replay --confidence energy --device 0
    
    # Multi-GPU
    python main.py -d mmea -m tbn_replay -g 0 1
"""

import argparse
import json
import os
import socket
import datetime
import uuid
import warnings
import torch
import wandb

from trainer import train

# 불필요한 경고 숨김 (출력 정돈)
warnings.filterwarnings("ignore")


def main():
    # ------------- 1) 인자 파싱 -------------
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, help='데이터셋 이름 (예: mmea)')
    parser.add_argument('-m', '--model_name', type=str, required=True, help='모델/실험 이름 (exps/exp_<name>.json)')
    parser.add_argument('-f', '--fusion_type', type=str, 
                        choices=['concat', 'imu_cosine_gate', 'imu_euclidean_gate', 'imu_kl_gate', 'imu_entropy_gate', 'hierarchical_gating', 'hierarchical_concat', 'auxiliary_head', 'auxiliary_head_v2', 'auxiliary_head_v2_3', 'auxiliary_head_v2_4', 'auxiliary_head_v2_5','auxiliary_head_v2_6', 'auxiliary_head_v2_7', 'auxiliary_head_v2_8', 'auxiliary_head_v2_9', 'auxiliary_head_v2_10', 'gated_cross_modal', 'attention', 'cross_attention'],
                        help='Fusion 방법 선택 (JSON 설정을 덮어씀)')
    parser.add_argument('-c', '--confidence', type=str, 
                        choices=['entropy', 'max_prob', 'energy', 'margin', 'variance', 'doctor'],
                        help='Confidence 계산 방법 (auxiliary_head_v2_7/8/9에서 사용, JSON 설정을 덮어씀)')
    parser.add_argument('-g', '--device', type=int, nargs='+',
                        help='GPU device ID(s) 선택 (예: --device 0 또는 -g 0, JSON 설정을 덮어씀)')
    parser.add_argument('--wandb_project', type=str, default='MMEA-OWCL_hj_test')
    parser.add_argument('--wandb_entity', type=str, default='mmea-owcl')
    parser.add_argument('--debug_mode', action='store_true',
                        help='디버그 모드: 학습 스텝 축소 + W&B 비활성')
    args, _ = parser.parse_known_args()

    # ------------- 2) 설정 JSON 로드 (exps/exp_<model>.json) -------------
    config_path = os.path.join("exps", f"exp_{args.dataset}_{args.model_name}.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # ------------- 3) argparse 값이 JSON을 덮어씀 (우선순위: JSON < argparse) -------------
    # confidence -> confidence_method로 매핑
    args_dict = vars(args)
    if args_dict.get('confidence') is not None:
        args_dict['confidence_method'] = args_dict.pop('confidence')
    
    # device 처리: list로 변환 (JSON 형식과 일치)
    if args_dict.get('device') is not None:
        # Command line에서 --device 0 또는 --device 0 1 형태로 입력받음
        # 이미 list 형태이므로 그대로 사용
        pass
    
    config.update(args_dict)

    # ------------- 4) 런 메타 정보 주입 (run_id/시간/호스트/GPU/W&B사용여부) -------------
    config['run_id'] = str(uuid.uuid4()).split('-')[0]
    config['timestamp'] = str(datetime.datetime.now())
    config['host'] = socket.gethostname()
    config['gpu_name'] = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
    config['use_wandb'] = bool(config.get('wandb_project') and config.get('wandb_entity'))

    # ------------- 5) 디버그 모드 처리 (W&B 끄기) -------------
    if config.get('debug_mode'):
        print('Debug mode enabled: running only a few forward steps per epoch with W&B disabled.')
        config['use_wandb'] = 0

    # ------------- 6) W&B 초기화 (스윕 값이 최종 덮어씀) -------------
    if config.get('use_wandb'):
        # mode에 따라 다른 wandb project 사용
        if config.get('mode') == 'eval':
            wandb_project = 'Experimental Results on the MMEA-OWCL (Evaluation CL & OOD)'
        else:
            wandb_project = config.get('wandb_project', 'MMEA-OWCL')
        
        wandb.init(
            project=wandb_project,
            entity=config['wandb_entity'],
            name=f"{config['model_name']}_{config['run_id']}",
            config=config
        )
        # 최종 설정(스윕)으로 덮어쓰기: JSON < argparse < W&B
        config.update(dict(wandb.config))

    # ------------- 7) init_cls 보정 및 검증 -------------
    # init_cls가 없거나 increment와 다르면 increment로 맞춤 (wandb 업데이트 후에도 적용)
    if config.get('init_cls') is None or config['init_cls'] != config['increment']:
        if config.get('init_cls') is not None:
            print(f"⚠️  init_cls ({config['init_cls']}) != increment ({config['increment']})")
            print(f"   → init_cls를 increment 값으로 자동 보정: {config['increment']}")
        config['init_cls'] = config['increment']
        
        # wandb가 활성화된 경우 wandb config도 업데이트
        if config.get('use_wandb', False) and 'wandb' in globals():
            wandb.config.update({'init_cls': config['increment']}, allow_val_change=True)

    # ------------- 8) 실험 요약 출력 -------------
    print("=" * 60)
    print("🚀 Multi-Modal Open World Continual Learning")
    print("=" * 60)
    print(f"✓ Dataset      : {config.get('dataset')}")
    print(f"✓ Model        : {config.get('model_name')}")
    print(f"✓ Fusion Type  : {config.get('fusion_type')}")
    print(f"✓ Modalities   : {config.get('modality')}")
    print(f"✓ Tasks        : Initial {config.get('init_cls')} + {config.get('increment')} each increment")
    print(f"✓ Confidence   : {config.get('confidence_method', 'max_prob (default)')}")
    print(f"✓ Device(s)    : {config.get('device', [0])}")
    print(f"✓ OOD Methods  : {config.get('ood_methods')}")
    print(f"✓ Use W&B      : {bool(config.get('use_wandb'))}")
    print("=" * 60)

    # ------------- 9) 학습 시작 -------------
    train(config)


if __name__ == '__main__':
    main()
