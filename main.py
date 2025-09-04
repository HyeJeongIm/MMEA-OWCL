"""
Multi-Modal Open World Continual Learning (MMOWCL)

Usage:
    python main.py -d mmea -m ewc_tbn_concat
    python main.py -d mmea -m ewc_tbn_concat --debug_mode
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
    parser.add_argument('--wandb_project', type=str, default='MMEA-OWCL')
    parser.add_argument('--wandb_entity', type=str, default='mmea-owcl')
    parser.add_argument('--debug_mode', action='store_true',
                        help='디버그 모드: 학습 스텝 축소 + W&B 비활성')
    args, _ = parser.parse_known_args()

    # ------------- 2) 설정 JSON 로드 (exps/exp_<model>.json) -------------
    config_path = os.path.join("exps", f"exp_{args.model_name}.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # ------------- 3) argparse 값이 JSON을 덮어씀 (우선순위: JSON < argparse) -------------
    config.update(vars(args))

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
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            name=f"{config['model_name']}_{config['run_id']}",
            config=config
        )
        # 최종 설정(스윕)으로 덮어쓰기: JSON < argparse < W&B
        config.update(dict(wandb.config))

    # ------------- 7) init_cls 보정 및 검증 -------------
    if config.get('init_cls') is None:
        config['init_cls'] = config['increment']
    assert config['init_cls'] == config['increment'], "init_cls and increment need to be same"

    # ------------- 8) 실험 요약 출력 -------------
    print("=" * 60)
    print("🚀 Multi-Modal Open World Continual Learning")
    print("=" * 60)
    print(f"✓ Dataset      : {config.get('dataset')}")
    print(f"✓ Model        : {config.get('model_name')}")
    print(f"✓ Modalities   : {config.get('modality')}")
    print(f"✓ Tasks        : Initial {config.get('init_cls')} + {config.get('increment')} each increment")
    print(f"✓ OOD Methods  : {config.get('ood_methods')}")
    print(f"✓ Use W&B      : {bool(config.get('use_wandb'))}")
    print("=" * 60)

    # ------------- 9) 학습 시작 -------------
    train(config)


if __name__ == '__main__':
    main()
