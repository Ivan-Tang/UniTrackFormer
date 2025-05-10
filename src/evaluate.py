import torch
from torch.utils.data import DataLoader
from src.models import UniTrackFormer
from src.dataset import TrackMLDataset
from src.losses import LossModule
import pandas as pd

def evaluate_model(model, dataloader, loss_fn, device='cpu'):
    model.eval()  # 设置模型为评估模式
    total_loss = 0

    with torch.no_grad():  # 禁用梯度计算
        for batch in dataloader:
            # 获取输入和标签
            X = batch['X'].squeeze(0).to(device)
            mask_label = batch['mask_labels'].squeeze(0).to(device)
            track_label = batch['track_labels'].squeeze(0).to(device)
            param_label = batch['track_params'].squeeze(0).to(device)

            # Forward pass
            out = model(X)

            # Compute loss
            loss_dict = loss_fn(
                track_logits=out['track_logits'],
                hit_masks=out['hit_assignment'],
                track_props=out['track_properties'],
                target_cls=track_label,
                target_masks=mask_label,
                target_props=param_label
            )
            total_loss += loss_dict['total'].item()

    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Done - Avg Loss: {avg_loss:.4f}")
    return avg_loss

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else: return 'cpu'

if __name__ == '__main__':
    # 加载数据集
    data_dir = 'data/validation_sample/test'
    detectors = pd.read_csv('data/detectors.csv')
    dataset = TrackMLDataset(data_dir=data_dir, detectors_df=detectors)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 加载模型
    model = UniTrackFormer(input_dim=dataset.feature_dim)
    checkpoint = torch.load('checkpoints/unitrackformer_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载损失函数
    loss_fn = LossModule()

    # 设置设备
    device = get_device()
    model = model.to(device)

    # 评估模型
    evaluate_model(model, loader, loss_fn, device)
