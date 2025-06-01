#!/usr/bin/env python3
"""
TrackML TrackFormer 高级可视化模块

功能：
1. 事件数据的多维度可视化
2. 模型预测结果的可视化对比
3. 轨迹重建质量分析
4. 交互式可视化（可选）
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.colors import ListedColormap
import torch

# 设置matplotlib样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TrackMLVisualizer:
    def __init__(self, save_dir='results'):
        """初始化可视化器"""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置字体以支持中文
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
    def plot_event_overview(self, hits, truth=None, save_name='event_overview'):
        """绘制事件总览：hits分布和基本统计"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 3D分布
        ax1 = fig.add_subplot(2, 4, 1, projection='3d')
        ax1.scatter(hits['x'], hits['y'], hits['z'], s=0.5, alpha=0.6)
        ax1.set_title('3D Hits Distribution')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        
        # 2. XY投影
        ax2 = fig.add_subplot(2, 4, 2)
        ax2.scatter(hits['x'], hits['y'], s=0.5, alpha=0.6)
        ax2.set_title('XY Projection')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.axis('equal')
        
        # 3. RZ投影
        ax3 = fig.add_subplot(2, 4, 3)
        r = np.sqrt(hits['x']**2 + hits['y']**2)
        ax3.scatter(hits['z'], r, s=0.5, alpha=0.6)
        ax3.set_title('RZ Projection')
        ax3.set_xlabel('Z (mm)')
        ax3.set_ylabel('R (mm)')
        
        # 4. XZ投影
        ax4 = fig.add_subplot(2, 4, 4)
        ax4.scatter(hits['x'], hits['z'], s=0.5, alpha=0.6)
        ax4.set_title('XZ Projection')
        ax4.set_xlabel('X (mm)')
        ax4.set_ylabel('Z (mm)')
        
        # 5. Volume分布
        ax5 = fig.add_subplot(2, 4, 5)
        volume_counts = hits['volume_id'].value_counts().sort_index()
        ax5.bar(volume_counts.index, volume_counts.values)
        ax5.set_title('Hits per Volume')
        ax5.set_xlabel('Volume ID')
        ax5.set_ylabel('Number of Hits')
        
        # 6. Layer分布
        ax6 = fig.add_subplot(2, 4, 6)
        layer_counts = hits['layer_id'].value_counts().sort_index()
        ax6.bar(layer_counts.index, layer_counts.values)
        ax6.set_title('Hits per Layer')
        ax6.set_xlabel('Layer ID')
        ax6.set_ylabel('Number of Hits')
        
        # 7. 径向分布
        ax7 = fig.add_subplot(2, 4, 7)
        ax7.hist(r, bins=50, alpha=0.7, edgecolor='black')
        ax7.set_title('Radial Distribution')
        ax7.set_xlabel('R (mm)')
        ax7.set_ylabel('Number of Hits')
        
        # 8. Z分布
        ax8 = fig.add_subplot(2, 4, 8)
        ax8.hist(hits['z'], bins=50, alpha=0.7, edgecolor='black')
        ax8.set_title('Z Distribution')
        ax8.set_xlabel('Z (mm)')
        ax8.set_ylabel('Number of Hits')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{save_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 如果有truth数据，绘制粒子统计
        if truth is not None:
            self._plot_particle_statistics(truth, save_name)
    
    def _plot_particle_statistics(self, truth, save_name):
        """绘制粒子统计信息"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. 粒子数量分布
        particle_hits = truth.groupby('particle_id').size()
        particle_hits = particle_hits[particle_hits.index != 0]  # 排除噪声
        
        axes[0, 0].hist(particle_hits.values, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Hits per Particle Distribution')
        axes[0, 0].set_xlabel('Number of Hits')
        axes[0, 0].set_ylabel('Number of Particles')
        
        # 2. 动量分布
        axes[0, 1].hist(np.sqrt(truth['tpx']**2 + truth['tpy']**2 + truth['tpz']**2), 
                       bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Momentum Distribution')
        axes[0, 1].set_xlabel('|p| (GeV/c)')
        axes[0, 1].set_ylabel('Number of Hits')
        
        # 3. 横向动量分布
        pt = np.sqrt(truth['tpx']**2 + truth['tpy']**2)
        axes[0, 2].hist(pt, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Transverse Momentum Distribution')
        axes[0, 2].set_xlabel('pt (GeV/c)')
        axes[0, 2].set_ylabel('Number of Hits')
        
        # 4. eta分布
        p = np.sqrt(truth['tpx']**2 + truth['tpy']**2 + truth['tpz']**2)
        eta = np.arctanh(truth['tpz'] / p)
        eta = eta[~np.isnan(eta) & ~np.isinf(eta)]
        axes[1, 0].hist(eta, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Pseudorapidity Distribution')
        axes[1, 0].set_xlabel('η')
        axes[1, 0].set_ylabel('Number of Hits')
        
        # 5. phi分布
        phi = np.arctan2(truth['tpy'], truth['tpx'])
        axes[1, 1].hist(phi, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Azimuthal Angle Distribution')
        axes[1, 1].set_xlabel('φ (rad)')
        axes[1, 1].set_ylabel('Number of Hits')
        
        # 6. 权重分布
        axes[1, 2].hist(truth['weight'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('Weight Distribution')
        axes[1, 2].set_xlabel('Weight')
        axes[1, 2].set_ylabel('Number of Hits')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{save_name}_particle_stats.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_track_comparison(self, hits, true_track_ids, pred_track_ids, 
                             event_id, max_tracks=20):
        """对比真实轨迹和预测轨迹"""
        fig = plt.figure(figsize=(24, 16))
        
        # 准备颜色
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        # 1. 真实轨迹 - 3D
        ax1 = fig.add_subplot(2, 4, 1, projection='3d')
        unique_true_ids = np.unique(true_track_ids)
        unique_true_ids = unique_true_ids[unique_true_ids != 0][:max_tracks]
        
        for i, track_id in enumerate(unique_true_ids):
            mask = true_track_ids == track_id
            track_hits = hits[mask]
            if len(track_hits) > 0:
                ax1.scatter(track_hits['x'], track_hits['y'], track_hits['z'],
                           s=2, alpha=0.8, c=[colors[i % len(colors)]], 
                           label=f'T{track_id}')
        ax1.set_title('Ground Truth Tracks (3D)')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        
        # 2. 预测轨迹 - 3D
        ax2 = fig.add_subplot(2, 4, 2, projection='3d')
        unique_pred_ids = np.unique(pred_track_ids)
        unique_pred_ids = unique_pred_ids[unique_pred_ids != 0][:max_tracks]
        
        for i, track_id in enumerate(unique_pred_ids):
            mask = pred_track_ids == track_id
            track_hits = hits[mask]
            if len(track_hits) > 0:
                ax2.scatter(track_hits['x'], track_hits['y'], track_hits['z'],
                           s=2, alpha=0.8, c=[colors[i % len(colors)]], 
                           label=f'P{track_id}')
        ax2.set_title('Predicted Tracks (3D)')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_zlabel('Z (mm)')
        
        # 3. 真实轨迹 - RZ
        ax3 = fig.add_subplot(2, 4, 3)
        r = np.sqrt(hits['x']**2 + hits['y']**2)
        for i, track_id in enumerate(unique_true_ids):
            mask = true_track_ids == track_id
            if np.any(mask):
                ax3.scatter(hits[mask]['z'], r[mask], s=2, alpha=0.8, 
                           c=[colors[i % len(colors)]])
        ax3.set_title('Ground Truth (RZ)')
        ax3.set_xlabel('Z (mm)')
        ax3.set_ylabel('R (mm)')
        
        # 4. 预测轨迹 - RZ
        ax4 = fig.add_subplot(2, 4, 4)
        for i, track_id in enumerate(unique_pred_ids):
            mask = pred_track_ids == track_id
            if np.any(mask):
                ax4.scatter(hits[mask]['z'], r[mask], s=2, alpha=0.8, 
                           c=[colors[i % len(colors)]])
        ax4.set_title('Predicted (RZ)')
        ax4.set_xlabel('Z (mm)')
        ax4.set_ylabel('R (mm)')
        
        # 5. 轨迹长度对比
        ax5 = fig.add_subplot(2, 4, 5)
        true_lengths = [np.sum(true_track_ids == tid) for tid in unique_true_ids]
        pred_lengths = [np.sum(pred_track_ids == tid) for tid in unique_pred_ids]
        
        ax5.hist(true_lengths, bins=20, alpha=0.7, label='Ground Truth', 
                color='blue', edgecolor='black')
        ax5.hist(pred_lengths, bins=20, alpha=0.7, label='Predicted', 
                color='red', edgecolor='black')
        ax5.set_title('Track Length Distribution')
        ax5.set_xlabel('Number of Hits')
        ax5.set_ylabel('Number of Tracks')
        ax5.legend()
        
        # 6. 匹配矩阵热图
        ax6 = fig.add_subplot(2, 4, 6)
        self._plot_matching_matrix(true_track_ids, pred_track_ids, ax6)
        
        # 7. 误分类分析
        ax7 = fig.add_subplot(2, 4, 7)
        self._plot_classification_analysis(true_track_ids, pred_track_ids, ax7)
        
        # 8. 性能摘要
        ax8 = fig.add_subplot(2, 4, 8)
        self._plot_performance_summary(true_track_ids, pred_track_ids, ax8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{event_id}_track_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_matching_matrix(self, true_track_ids, pred_track_ids, ax):
        """绘制轨迹匹配矩阵"""
        unique_true = np.unique(true_track_ids)
        unique_pred = np.unique(pred_track_ids)
        
        # 限制显示的轨迹数量
        unique_true = unique_true[unique_true != 0][:10]
        unique_pred = unique_pred[unique_pred != 0][:10]
        
        # 创建匹配矩阵
        match_matrix = np.zeros((len(unique_true), len(unique_pred)))
        
        for i, true_id in enumerate(unique_true):
            for j, pred_id in enumerate(unique_pred):
                # 计算重叠度
                true_mask = true_track_ids == true_id
                pred_mask = pred_track_ids == pred_id
                overlap = np.sum(true_mask & pred_mask)
                match_matrix[i, j] = overlap
        
        # 归一化
        row_sums = np.sum(match_matrix, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        match_matrix = match_matrix / row_sums
        
        im = ax.imshow(match_matrix, cmap='Blues', aspect='auto')
        ax.set_title('Track Matching Matrix')
        ax.set_xlabel('Predicted Track ID')
        ax.set_ylabel('True Track ID')
        ax.set_xticks(range(len(unique_pred)))
        ax.set_xticklabels(unique_pred)
        ax.set_yticks(range(len(unique_true)))
        ax.set_yticklabels(unique_true)
        
        # 添加数值标注
        for i in range(len(unique_true)):
            for j in range(len(unique_pred)):
                text = ax.text(j, i, f'{match_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
    
    def _plot_classification_analysis(self, true_track_ids, pred_track_ids, ax):
        """分析分类结果"""
        # 计算混淆矩阵的统计
        n_true_tracks = len(np.unique(true_track_ids)) - (1 if 0 in true_track_ids else 0)
        n_pred_tracks = len(np.unique(pred_track_ids)) - (1 if 0 in pred_track_ids else 0)
        
        # 计算未分配的hits
        true_unassigned = np.sum(true_track_ids == 0)
        pred_unassigned = np.sum(pred_track_ids == 0)
        
        categories = ['True Tracks', 'Pred Tracks', 'True Unassigned', 'Pred Unassigned']
        values = [n_true_tracks, n_pred_tracks, true_unassigned, pred_unassigned]
        colors = ['lightblue', 'lightcoral', 'lightgray', 'darkgray']
        
        bars = ax.bar(categories, values, color=colors, edgecolor='black')
        ax.set_title('Classification Summary')
        ax.set_ylabel('Count')
        
        # 添加数值标注
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(values),
                   f'{value}', ha='center', va='bottom')
    
    def _plot_performance_summary(self, true_track_ids, pred_track_ids, ax):
        """绘制性能摘要"""
        from src.metric import evaluate_metrics
        
        try:
            metrics = evaluate_metrics(true_track_ids, pred_track_ids)
            
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = ax.bar(metric_names, metric_values, 
                         color=['skyblue', 'lightcoral', 'lightgreen'][:len(metric_names)],
                         edgecolor='black')
            
            ax.set_title('Performance Metrics')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # 添加数值标注
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Metrics calculation failed:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Metrics (Error)')
    
    def plot_training_progress(self, train_losses, val_losses=None, save_name='training_progress'):
        """绘制训练进度"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        epochs = range(1, len(train_losses) + 1)
        axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if val_losses:
            axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_title('Training Progress')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 损失改善率
        if len(train_losses) > 1:
            train_improvement = np.diff(train_losses)
            axes[1].plot(epochs[1:], train_improvement, 'b-', 
                        label='Training Loss Change', linewidth=2)
            if val_losses and len(val_losses) > 1:
                val_improvement = np.diff(val_losses)
                axes[1].plot(epochs[1:], val_improvement, 'r-', 
                            label='Validation Loss Change', linewidth=2)
            axes[1].set_title('Loss Improvement per Epoch')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss Change')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{save_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_event_report(self, event_id, hits, truth, predictions, metrics):
        """创建完整的事件分析报告"""
        print(f"Creating comprehensive report for {event_id}...")
        
        # 1. 事件总览
        self.plot_event_overview(hits, truth, f'{event_id}_overview')
        
        # 2. 后处理预测结果
        pred_track_ids = self._postprocess_predictions(predictions)
        
        # 3. 创建真实轨迹ID
        true_track_ids = self._create_true_track_ids(truth, hits)
        
        # 4. 轨迹对比
        self.plot_track_comparison(hits, true_track_ids, pred_track_ids, event_id)
        
        # 5. 保存文本报告
        self._save_text_report(event_id, hits, truth, predictions, metrics)
        
        print(f"Report completed for {event_id}")
    
    def _postprocess_predictions(self, predictions, threshold=0.5):
        """后处理预测结果"""
        output = predictions['predictions']
        
        if 'track_logits' not in output or output['track_logits'].numel() == 0:
            return np.zeros(predictions['input_features'].shape[0], dtype=int)
        
        track_logits = torch.sigmoid(output['track_logits']).cpu().numpy()
        hit_assignments = torch.sigmoid(output['hit_assignment']).cpu().numpy()
        
        n_hits = predictions['input_features'].shape[0]
        predicted_track_ids = np.zeros(n_hits, dtype=int)
        
        for query_idx, (track_score, hit_scores) in enumerate(zip(track_logits, hit_assignments)):
            if track_score > threshold:
                assigned_hits = hit_scores > threshold
                predicted_track_ids[assigned_hits] = query_idx + 1
        
        return predicted_track_ids
    
    def _create_true_track_ids(self, truth, hits):
        """从truth数据创建轨迹ID"""
        true_track_ids = np.zeros(len(hits), dtype=int)
        
        if truth is not None:
            hit_to_particle = dict(zip(truth['hit_id'], truth['particle_id']))
            for i, hit_id in enumerate(hits['hit_id']):
                particle_id = hit_to_particle.get(hit_id, 0)
                true_track_ids[i] = particle_id
        
        return true_track_ids
    
    def _save_text_report(self, event_id, hits, truth, predictions, metrics):
        """保存文本格式的详细报告"""
        report_path = os.path.join(self.save_dir, f'{event_id}_detailed_report.txt')
        
        with open(report_path, 'w') as f:
            f.write(f"TrackML Event Analysis Report\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Event ID: {event_id}\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 基本统计
            f.write(f"Basic Statistics:\n")
            f.write(f"- Total Hits: {len(hits)}\n")
            f.write(f"- Unique Volumes: {hits['volume_id'].nunique()}\n")
            f.write(f"- Unique Layers: {hits['layer_id'].nunique()}\n")
            
            if truth is not None:
                unique_particles = truth[truth['particle_id'] != 0]['particle_id'].nunique()
                f.write(f"- True Particles: {unique_particles}\n")
                avg_hits_per_particle = len(truth[truth['particle_id'] != 0]) / unique_particles
                f.write(f"- Avg Hits per Particle: {avg_hits_per_particle:.2f}\n")
            
            # 预测统计
            pred_track_ids = self._postprocess_predictions(predictions)
            unique_pred_tracks = len(np.unique(pred_track_ids)) - (1 if 0 in pred_track_ids else 0)
            f.write(f"- Predicted Tracks: {unique_pred_tracks}\n")
            
            # 性能指标
            f.write(f"\nPerformance Metrics:\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"- {metric_name.capitalize()}: {metric_value:.4f}\n")
            
            # 模型输出分析
            f.write(f"\nModel Output Analysis:\n")
            output = predictions['predictions']
            if 'track_logits' in output and output['track_logits'].numel() > 0:
                track_scores = torch.sigmoid(output['track_logits']).cpu().numpy()
                f.write(f"- Track Confidence (mean±std): {track_scores.mean():.3f}±{track_scores.std():.3f}\n")
                f.write(f"- Track Confidence (min-max): {track_scores.min():.3f}-{track_scores.max():.3f}\n")
                f.write(f"- Tracks above 0.5 threshold: {np.sum(track_scores > 0.5)}/{len(track_scores)}\n")
            
            f.write(f"\nNote: Visualizations saved as {event_id}_*.png\n")
        
        print(f"Detailed report saved to {report_path}")


# 便捷函数
def quick_visualize_event(event_data_dir, event_id, model_path=None, save_dir='results'):
    """快速可视化事件（带可选的模型预测）"""
    visualizer = TrackMLVisualizer(save_dir)
    
    # 加载事件数据
    from src.visual import load_event_data
    hits, truth = load_event_data(event_data_dir, event_id)
    
    # 基本可视化
    visualizer.plot_event_overview(hits, truth, f'{event_id}_quick')
    
    # 如果提供了模型，进行预测和对比
    if model_path and os.path.exists(model_path):
        try:
            from test_and_predict import TrackMLPredictor
            predictor = TrackMLPredictor(model_path)
            predictions = predictor.predict_event(event_data_dir, event_id)
            metrics = predictor.evaluate_predictions(predictions)
            
            # 创建完整报告
            visualizer.create_event_report(event_id, hits, truth, predictions, metrics)
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            print("Continuing with basic visualization only...")
    
    print(f"Visualization completed for {event_id}")


if __name__ == '__main__':
    # 示例用法
    quick_visualize_event(
        event_data_dir='data/train_sample',
        event_id='event000001000',
        model_path='checkpoints/best_model.pth',
        save_dir='results'
    )
