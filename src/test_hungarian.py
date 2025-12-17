#!/usr/bin/env python3
"""
测试匈牙利匹配算法的简化脚本
"""

import torch
import numpy as np
from losses import LossModule
from utils import get_device

def test_hungarian_matching():
    """测试匈牙利匹配功能"""
    device = get_device()
    
    # 创建模拟数据
    n_queries = 100
    n_tracks = 64
    n_hits = 5000
    
    # 模型输出（模拟）
    track_logits = torch.randn(n_queries, device=device)
    hit_assignment = torch.randn(n_queries, n_hits, device=device)
    track_properties = torch.randn(n_queries, 6, device=device)  # 6个参数
    
    # 真实标签
    target_cls = torch.ones(n_tracks, device=device)  # 所有都是真实轨道
    target_masks = torch.randint(0, 2, (n_tracks, n_hits), device=device).float()
    target_props = torch.randn(n_tracks, 6, device=device)  # 6个参数
    
    print(f"模型输出维度:")
    print(f"  track_logits: {track_logits.shape}")
    print(f"  hit_assignment: {hit_assignment.shape}")
    print(f"  track_properties: {track_properties.shape}")
    
    print(f"\n真实标签维度:")
    print(f"  target_cls: {target_cls.shape}")
    print(f"  target_masks: {target_masks.shape}")
    print(f"  target_props: {target_props.shape}")
    
    # 创建损失函数
    loss_fn = LossModule(alpha=1, beta=1, gamma=1)
    
    print(f"\n计算损失...")
    try:
        loss_dict = loss_fn(
            track_logits=track_logits,
            hit_masks=hit_assignment,
            track_props=track_properties,
            target_cls=target_cls,
            target_masks=target_masks,
            target_props=target_props
        )
        
        print(f"损失计算成功!")
        print(f"  总损失: {loss_dict['total'].item():.4f}")
        print(f"  分类损失: {loss_dict['cls'].item():.4f}")
        print(f"  掩码损失: {loss_dict['mask'].item():.4f}")
        print(f"  参数损失: {loss_dict['param'].item():.4f}")
        
    except Exception as e:
        print(f"损失计算失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hungarian_matching()
