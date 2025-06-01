#!/usr/bin/env python3
"""
TrackML TrackFormer 完整测试和预测脚本

功能：
1. 加载训练好的模型
2. 对指定事件进行预测
3. 保存预测结果
4. 可视化预测结果并与真实轨迹对比
5. 计算评估指标

用法：
    python test_and_predict.py --event event000001000 --model checkpoints/best_model.pth
    python test_and_predict.py --test_dir data/test --save_submission
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset import TrackMLDataset
from src.trackformer import create_trackformer_600mev
from src.losses import LossModule
from src.metric import evaluate_metrics
from src.visual import (
    visualize_hits_3d, 
    visualize_hits_rz, 
    visualize_ground_truth,
    load_event_data
)


class TrackMLPredictor:
    def __init__(self, model_path, device='auto'):
        """初始化预测器"""
        self.device = self._get_device() if device == 'auto' else device
        self.model = None
        self.detectors = pd.read_csv('data/detectors.csv')
        self.load_model(model_path)
        
    def _get_device(self):
        """自动检测设备"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self, model_path):
        """加载训练好的模型（只支持新的TrackFormer架构）"""
        print(f"Loading model from {model_path} on device {self.device}")
        
        # 创建临时数据集来获取特征维度
        temp_event_ids = ['event000001000']  # 假设至少有这一个事件
        temp_dataset = TrackMLDataset('data/train_sample', self.detectors, temp_event_ids)
        
        # 创建新的TrackFormer模型
        self.model = create_trackformer_600mev(input_dim=temp_dataset.feature_dim)
        
        # 加载权重
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 检查是否为新TrackFormer架构
            if not any('hit_filter.encoder_layers' in key for key in state_dict.keys()):
                raise ValueError("Model checkpoint is from legacy UniTrackFormer architecture. "
                               "Please retrain with new TrackFormer architecture using main_train.py")
            
            self.model.load_state_dict(state_dict)
            print("Successfully loaded TrackFormer model")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure you have a trained TrackFormer model. Run 'python main_train.py' to train a new model.")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def predict_event(self, event_data_dir, event_id):
        """对单个事件进行预测"""
        print(f"Predicting event {event_id}...")
        
        # 创建数据集
        dataset = TrackMLDataset(event_data_dir, self.detectors, [event_id])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # 获取数据
        batch = next(iter(dataloader))
        X = batch['X'].squeeze(0).to(self.device)
        mask_labels = batch['mask_labels'].squeeze(0)
        track_labels = batch['track_labels'].squeeze(0)
        track_params = batch['track_params'].squeeze(0)
        
        # 提取坐标
        x, y, z = X[:, 1], X[:, 2], X[:, 3]
        r = torch.sqrt(x**2 + y**2)
        phi = torch.atan2(y, x)
        coords = torch.stack([r, phi, z], dim=1)
        
        # 模型预测
        with torch.no_grad():
            output = self.model(X, coords)
        
        return {
            'input_features': X.cpu(),
            'true_masks': mask_labels,
            'true_labels': track_labels,
            'true_params': track_params,
            'predictions': output
        }
    
    def postprocess_predictions(self, predictions, threshold=0.5):
        """后处理预测结果，转换为轨迹ID格式"""
        output = predictions['predictions']
        
        if 'track_logits' not in output or output['track_logits'].numel() == 0:
            print("No valid predictions found")
            return np.zeros(predictions['input_features'].shape[0], dtype=int)
        
        # 获取预测
        track_logits = torch.sigmoid(output['track_logits']).cpu().numpy()
        hit_assignments = torch.sigmoid(output['hit_assignment']).cpu().numpy()
        
        # 创建轨迹ID分配
        n_hits = predictions['input_features'].shape[0]
        predicted_track_ids = np.zeros(n_hits, dtype=int)
        
        # 对每个轨迹查询
        for query_idx, (track_score, hit_scores) in enumerate(zip(track_logits, hit_assignments)):
            if track_score > threshold:  # 轨迹存在
                # 找到属于该轨迹的hits
                assigned_hits = hit_scores > threshold
                predicted_track_ids[assigned_hits] = query_idx + 1
        
        return predicted_track_ids
    
    def save_predictions(self, event_id, predictions, save_dir='results'):
        """保存预测结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 后处理预测结果
        predicted_track_ids = self.postprocess_predictions(predictions)
        
        # 创建结果DataFrame
        n_hits = predictions['input_features'].shape[0]
        result_df = pd.DataFrame({
            'hit_id': range(1, n_hits + 1),
            'track_id': predicted_track_ids
        })
        
        # 保存预测结果
        output_path = os.path.join(save_dir, f'{event_id}_predictions.csv')
        result_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        return result_df
    
    def evaluate_predictions(self, predictions):
        """评估预测结果"""
        true_masks = predictions['true_masks'].numpy()
        true_labels = predictions['true_labels'].numpy()
        
        # 从真实标签创建轨迹ID
        true_track_ids = np.zeros(true_masks.shape[1], dtype=int)
        for track_idx, mask in enumerate(true_masks):
            if true_labels[track_idx] > 0:  # 有效轨迹
                true_track_ids[mask > 0.5] = track_idx + 1
        
        # 获取预测轨迹ID
        predicted_track_ids = self.postprocess_predictions(predictions)
        
        # 计算指标
        metrics = evaluate_metrics(true_track_ids, predicted_track_ids)
        
        print(f"Evaluation Metrics:")
        print(f"  Efficiency: {metrics['efficiency']:.3f}")
        print(f"  Fake Rate: {metrics['fake_rate']:.3f}")
        if 'purity' in metrics:
            print(f"  Purity: {metrics['purity']:.3f}")
        
        return metrics


def visualize_predictions(event_data_dir, event_id, predictions, save_dir='results'):
    """可视化预测结果"""
    print(f"Visualizing predictions for {event_id}...")
    
    # 加载原始数据
    hits, truth = load_event_data(event_data_dir, event_id)
    
    # 获取预测轨迹ID
    predictor = TrackMLPredictor.__new__(TrackMLPredictor)  # 临时实例
    predicted_track_ids = predictor.postprocess_predictions(predictions)
    
    # 创建可视化
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 原始hits分布
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(hits['x'], hits['y'], hits['z'], s=1, alpha=0.6, c='gray')
    ax1.set_title('Original Hits Distribution')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    
    # 2. 真实轨迹（如果有truth数据）
    if truth is not None:
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        # 按particle_id着色
        unique_pids = truth['particle_id'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_pids), 20)))
        
        for i, pid in enumerate(unique_pids[:20]):  # 只显示前20个轨迹
            if pid == 0:  # 跳过噪声
                continue
            track_hits = truth[truth['particle_id'] == pid]
            hit_coords = hits[hits['hit_id'].isin(track_hits['hit_id'])]
            if len(hit_coords) > 0:
                ax2.scatter(hit_coords['x'], hit_coords['y'], hit_coords['z'], 
                           s=2, alpha=0.8, c=[colors[i % len(colors)]], 
                           label=f'Track {pid}')
        
        ax2.set_title('Ground Truth Tracks')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_zlabel('Z (mm)')
    
    # 3. 预测轨迹
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    unique_track_ids = np.unique(predicted_track_ids)
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_track_ids), 20)))
    
    for i, track_id in enumerate(unique_track_ids):
        if track_id == 0:  # 跳过未分配的hits
            continue
        track_hits_mask = predicted_track_ids == track_id
        track_hits_coords = hits.iloc[track_hits_mask]
        if len(track_hits_coords) > 0:
            ax3.scatter(track_hits_coords['x'], track_hits_coords['y'], track_hits_coords['z'],
                       s=2, alpha=0.8, c=[colors[i % len(colors)]], 
                       label=f'Pred Track {track_id}')
    
    ax3.set_title('Predicted Tracks')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_zlabel('Z (mm)')
    
    # 4-6. RZ投影视图
    ax4 = fig.add_subplot(2, 3, 4)
    r = np.sqrt(hits['x']**2 + hits['y']**2)
    ax4.scatter(hits['z'], r, s=1, alpha=0.6, c='gray')
    ax4.set_title('Original Hits (RZ plane)')
    ax4.set_xlabel('Z (mm)')
    ax4.set_ylabel('R (mm)')
    
    if truth is not None:
        ax5 = fig.add_subplot(2, 3, 5)
        for i, pid in enumerate(unique_pids[:20]):
            if pid == 0:
                continue
            track_hits = truth[truth['particle_id'] == pid]
            hit_coords = hits[hits['hit_id'].isin(track_hits['hit_id'])]
            if len(hit_coords) > 0:
                r_track = np.sqrt(hit_coords['x']**2 + hit_coords['y']**2)
                ax5.scatter(hit_coords['z'], r_track, s=2, alpha=0.8, 
                           c=[colors[i % len(colors)]])
        ax5.set_title('Ground Truth (RZ plane)')
        ax5.set_xlabel('Z (mm)')
        ax5.set_ylabel('R (mm)')
    
    ax6 = fig.add_subplot(2, 3, 6)
    for i, track_id in enumerate(unique_track_ids):
        if track_id == 0:
            continue
        track_hits_mask = predicted_track_ids == track_id
        track_hits_coords = hits.iloc[track_hits_mask]
        if len(track_hits_coords) > 0:
            r_track = np.sqrt(track_hits_coords['x']**2 + track_hits_coords['y']**2)
            ax6.scatter(track_hits_coords['z'], r_track, s=2, alpha=0.8,
                       c=[colors[i % len(colors)]])
    ax6.set_title('Predictions (RZ plane)')
    ax6.set_xlabel('Z (mm)')
    ax6.set_ylabel('R (mm)')
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{event_id}_prediction_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='TrackML TrackFormer Testing and Prediction')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--event', type=str, default='event000001000',
                       help='Event ID to predict')
    parser.add_argument('--data_dir', type=str, default='data/train_sample',
                       help='Directory containing event data')
    parser.add_argument('--test_dir', type=str, default=None,
                       help='Directory for test data (if different from data_dir)')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cpu, cuda, mps')
    parser.add_argument('--save_submission', action='store_true',
                       help='Save submission format for Kaggle')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Create visualizations')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 初始化预测器
    predictor = TrackMLPredictor(args.model, args.device)
    
    if args.test_dir and args.save_submission:
        # 批量处理测试集
        print("Processing test set for submission...")
        test_events = sorted([f.split('-')[0] for f in os.listdir(args.test_dir) 
                             if f.endswith('-hits.csv')])
        
        all_predictions = []
        for event_id in tqdm(test_events, desc="Processing events"):
            try:
                predictions = predictor.predict_event(args.test_dir, event_id)
                result_df = predictor.save_predictions(event_id, predictions, args.save_dir)
                all_predictions.append(result_df)
            except Exception as e:
                print(f"Error processing {event_id}: {e}")
                continue
        
        # 合并所有预测结果
        if all_predictions:
            submission_df = pd.concat(all_predictions, ignore_index=True)
            submission_path = os.path.join(args.save_dir, 'submission.csv')
            submission_df.to_csv(submission_path, index=False)
            print(f"Submission saved to {submission_path}")
    
    else:
        # 单个事件预测
        data_dir = args.test_dir if args.test_dir else args.data_dir
        
        # 进行预测
        predictions = predictor.predict_event(data_dir, args.event)
        
        # 保存预测结果
        predictor.save_predictions(args.event, predictions, args.save_dir)
        
        # 评估预测结果（如果有真实标签）
        if args.data_dir in ['data/train_sample', 'data/train_10_events']:
            metrics = predictor.evaluate_predictions(predictions)
            
            # 保存评估结果
            metrics_path = os.path.join(args.save_dir, f'{args.event}_metrics.txt')
            with open(metrics_path, 'w') as f:
                f.write(f"Event: {args.event}\n")
                f.write(f"Efficiency: {metrics['efficiency']:.3f}\n")
                f.write(f"Fake Rate: {metrics['fake_rate']:.3f}\n")
                if 'purity' in metrics:
                    f.write(f"Purity: {metrics['purity']:.3f}\n")
            print(f"Metrics saved to {metrics_path}")
        
        # 创建可视化
        if args.visualize:
            visualize_predictions(data_dir, args.event, predictions, args.save_dir)
    
    print("Testing and prediction completed!")


if __name__ == '__main__':
    main()
