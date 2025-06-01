#!/usr/bin/env python3
"""
TrackML TrackFormer 完整演示脚本

这个脚本演示了从训练到测试、预测、可视化的完整流程。

功能流程：
1. 数据探索和可视化
2. 模型训练（可选，如果没有现成模型）
3. 模型测试和预测
4. 结果可视化和分析
5. 性能评估

用法：
    python demo_complete_pipeline.py [--train] [--event event000001000]
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# 导入所有必要的模块
from src.dataset import TrackMLDataset
from src.trackformer import create_trackformer_600mev
from src.trainer import main as train_main
from test_and_predict import TrackMLPredictor, visualize_predictions
from advanced_visualization import TrackMLVisualizer, quick_visualize_event


def check_data_availability():
    """检查数据可用性"""
    data_paths = {
        'train_sample': 'data/train_sample',
        'test': 'data/test',
        'detectors': 'data/detectors.csv'
    }
    
    available_data = {}
    for name, path in data_paths.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                files = [f for f in os.listdir(path) if f.endswith('-hits.csv')]
                available_data[name] = len(files)
            else:
                available_data[name] = True
        else:
            available_data[name] = False
    
    return available_data


def check_model_availability():
    """检查预训练模型可用性"""
    model_paths = [
        'checkpoints/best_model.pth',
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            return path
    
    return None


def explore_data(event_id='event000001000'):
    """数据探索阶段"""
    print(f"\n{'='*60}")
    print(f"📊 DATA EXPLORATION PHASE")
    print(f"{'='*60}")
    
    # 检查数据
    data_status = check_data_availability()
    print(f"📁 Data Status:")
    for name, status in data_status.items():
        if isinstance(status, bool):
            print(f"   {name}: {'✅ Available' if status else '❌ Missing'}")
        else:
            print(f"   {name}: {'✅' if status > 0 else '❌'} {status} events")
    
    if not data_status['train_sample']:
        print("❌ No training data found. Please check data/train_sample directory.")
        return False
    
    # 加载和可视化示例事件
    try:
        print(f"\n🔍 Exploring event: {event_id}")
        
        # 创建可视化器
        visualizer = TrackMLVisualizer('results/exploration')
        
        # 加载事件数据
        from src.visual import load_event_data
        hits, truth = load_event_data('data/train_sample', event_id)
        
        print(f"   📈 Event Statistics:")
        print(f"      - Total hits: {len(hits):,}")
        print(f"      - Unique volumes: {hits['volume_id'].nunique()}")
        print(f"      - Unique layers: {hits['layer_id'].nunique()}")
        
        if truth is not None:
            unique_particles = truth[truth['particle_id'] != 0]['particle_id'].nunique()
            print(f"      - True particles: {unique_particles}")
        
        # 创建可视化
        visualizer.plot_event_overview(hits, truth, f'{event_id}_exploration')
        print(f"   📸 Visualizations saved to results/exploration/")
        
        return True
        
    except Exception as e:
        print(f"❌ Data exploration failed: {e}")
        return False


def train_or_load_model(force_train=False):
    """训练或加载模型"""
    print(f"\n{'='*60}")
    print(f"🧠 MODEL PREPARATION PHASE")
    print(f"{'='*60}")
    
    # 检查现有模型
    existing_model = check_model_availability()
    
    if existing_model and not force_train:
        print(f"✅ Found existing model: {existing_model}")
        print(f"   Skipping training. Use --train to force retraining.")
        return existing_model
    
    if force_train or existing_model is None:
        print(f"🏋️ Starting model training...")
        
        if existing_model is None:
            print(f"   No existing model found.")
        else:
            print(f"   Force retraining requested.")
        
        try:
            # 确保checkpoints目录存在
            os.makedirs('checkpoints', exist_ok=True)
            
            # 运行训练
            print(f"   🔄 Training TrackFormer model...")
            print(f"   ⏰ This may take several minutes to hours depending on your hardware...")
            
            # 调用训练主函数
            train_main()
            
            # 检查训练结果
            model_path = 'checkpoints/best_model.pth'
            if os.path.exists(model_path):
                print(f"   ✅ Training completed successfully!")
                print(f"   💾 Model saved to: {model_path}")
                return model_path
            else:
                print(f"   ❌ Training completed but model not found.")
                return None
                
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            return None
    
    return existing_model


def test_and_predict_model(model_path, event_id='event000001000'):
    """测试和预测阶段"""
    print(f"\n{'='*60}")
    print(f"🔮 TESTING AND PREDICTION PHASE")
    print(f"{'='*60}")
    
    try:
        print(f"🤖 Loading model: {model_path}")
        predictor = TrackMLPredictor(model_path)
        
        print(f"🎯 Predicting event: {event_id}")
        predictions = predictor.predict_event('data/train_sample', event_id)
        
        # 保存预测结果
        result_df = predictor.save_predictions(event_id, predictions, 'results/predictions')
        print(f"💾 Predictions saved")
        
        # 评估预测结果
        print(f"📊 Evaluating predictions...")
        metrics = predictor.evaluate_predictions(predictions)
        
        print(f"   📈 Performance Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"      - {metric_name.capitalize()}: {metric_value:.3f}")
        
        return predictions, metrics
        
    except Exception as e:
        print(f"❌ Testing and prediction failed: {e}")
        return None, None


def advanced_visualization_analysis(event_id, predictions, metrics):
    """高级可视化和分析"""
    print(f"\n{'='*60}")
    print(f"📈 ADVANCED VISUALIZATION PHASE")
    print(f"{'='*60}")
    
    try:
        print(f"🎨 Creating comprehensive visualizations...")
        
        # 创建高级可视化器
        visualizer = TrackMLVisualizer('results/analysis')
        
        # 加载事件数据
        from src.visual import load_event_data
        hits, truth = load_event_data('data/train_sample', event_id)
        
        # 创建完整报告
        visualizer.create_event_report(event_id, hits, truth, predictions, metrics)
        
        # 创建预测对比可视化
        visualize_predictions('data/train_sample', event_id, predictions, 'results/analysis')
        
        print(f"   📸 Comprehensive visualizations created")
        print(f"   📄 Detailed report saved")
        print(f"   📁 Results saved to results/analysis/")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced visualization failed: {e}")
        return False


def generate_summary_report():
    """生成总结报告"""
    print(f"\n{'='*60}")
    print(f"📋 SUMMARY REPORT")
    print(f"{'='*60}")
    
    # 收集所有结果文件
    result_dirs = ['results/exploration', 'results/predictions', 'results/analysis']
    
    print(f"📁 Generated Files:")
    total_files = 0
    
    for result_dir in result_dirs:
        if os.path.exists(result_dir):
            files = [f for f in os.listdir(result_dir) if f.endswith(('.png', '.csv', '.txt'))]
            if files:
                print(f"\n   📂 {result_dir}:")
                for file in sorted(files):
                    print(f"      - {file}")
                    total_files += 1
    
    print(f"\n📊 Summary:")
    print(f"   - Total files generated: {total_files}")
    print(f"   - Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 保存总结报告
    summary_path = 'results/pipeline_summary.txt'
    os.makedirs('results', exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write(f"TrackML TrackFormer Complete Pipeline Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Files Generated: {total_files}\n\n")
        
        for result_dir in result_dirs:
            if os.path.exists(result_dir):
                files = [f for f in os.listdir(result_dir) if f.endswith(('.png', '.csv', '.txt'))]
                if files:
                    f.write(f"{result_dir}:\n")
                    for file in sorted(files):
                        f.write(f"  - {file}\n")
                    f.write(f"\n")
    
    print(f"💾 Summary report saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='TrackML TrackFormer Complete Pipeline Demo')
    parser.add_argument('--train', action='store_true',
                       help='Force model training even if existing model found')
    parser.add_argument('--event', type=str, default='event000001000',
                       help='Event ID to analyze')
    parser.add_argument('--skip_exploration', action='store_true',
                       help='Skip data exploration phase')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training phase (use existing model only)')
    parser.add_argument('--skip_prediction', action='store_true',
                       help='Skip prediction phase')
    parser.add_argument('--skip_visualization', action='store_true',
                       help='Skip advanced visualization phase')
    
    args = parser.parse_args()
    
    print(f"🚀 TrackML TrackFormer Complete Pipeline Demo")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target event: {args.event}")
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    success_phases = []
    
    # Phase 1: 数据探索
    if not args.skip_exploration:
        if explore_data(args.event):
            success_phases.append("Data Exploration")
    
    # Phase 2: 模型训练/加载
    model_path = None
    if not args.skip_training:
        model_path = train_or_load_model(args.train)
        if model_path:
            success_phases.append("Model Preparation")
    else:
        model_path = check_model_availability()
        if model_path:
            print(f"✅ Using existing model: {model_path}")
    
    # Phase 3: 测试和预测
    predictions = None
    metrics = None
    if not args.skip_prediction and model_path:
        predictions, metrics = test_and_predict_model(model_path, args.event)
        if predictions is not None:
            success_phases.append("Testing and Prediction")
    
    # Phase 4: 高级可视化
    if not args.skip_visualization and predictions is not None:
        if advanced_visualization_analysis(args.event, predictions, metrics):
            success_phases.append("Advanced Visualization")
    
    # 生成总结报告
    generate_summary_report()
    
    # 最终总结
    print(f"\n{'='*60}")
    print(f"🎉 PIPELINE COMPLETION")
    print(f"{'='*60}")
    
    print(f"✅ Completed Phases: {len(success_phases)}")
    for phase in success_phases:
        print(f"   - {phase}")
    
    if len(success_phases) == 0:
        print(f"❌ No phases completed successfully")
        sys.exit(1)
    elif len(success_phases) < 4:
        print(f"⚠️  Some phases were skipped or failed")
    else:
        print(f"🎊 All phases completed successfully!")
    
    print(f"📁 All results saved to results/ directory")
    print(f"⏰ Total time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
