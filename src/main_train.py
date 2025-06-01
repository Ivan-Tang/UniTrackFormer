#!/usr/bin/env python3
"""
TrackML TrackFormer 训练脚本

用法:
    python main_train.py [--mode single|kfold] [--epochs 100] [--device auto]

模式说明:
    single: 单次训练验证 (默认)
    kfold:  K折交叉验证
"""

import argparse
import os
import pandas as pd
import torch

from src.trainer import main as single_train, train_kfold


def get_device():
    """自动检测最佳设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    parser = argparse.ArgumentParser(description='TrackML TrackFormer Training')
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'kfold'],
                       help='Training mode: single or k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cpu, cuda, mps')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of folds for k-fold training')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    print(f"训练模式: {args.mode}")
    print(f"训练轮数: {args.epochs}")
    
    # 确保目录存在
    os.makedirs('checkpoints', exist_ok=True)
    
    if args.mode == 'single':
        print("开始单次训练...")
        single_train()
    elif args.mode == 'kfold':
        print(f"开始 {args.folds} 折交叉验证训练...")
        
        # 准备数据
        data_dir = 'data/train_sample'
        detectors = pd.read_csv('data/detectors.csv')
        
        all_event_ids = sorted(
            set(
                fname.split('-')[0]
                for fname in os.listdir(data_dir) if fname.endswith('-hits.csv')
            )
        )
        
        train_kfold(
            data_dir=data_dir,
            detectors=detectors, 
            event_ids=all_event_ids,
            num_folds=args.folds,
            num_epochs=args.epochs,
            device=device
        )
    
    print("训练完成!")


if __name__ == '__main__':
    main()
