import torch
from torch.utils.data import DataLoader
from src.models import UniTrackFormer
from src.dataset import TrackMLDataset
from src.losses import LossModule
import pandas as pd

def train_one_epoch(model, loss_fn, dataloader, optimizer, device='cpu'):
    model.train()
    total_loss = 0

    def get_device():
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
        
    device = get_device()
    print(f"Training on {device}...")
    model = model.to(device)

    for i, batch in enumerate(dataloader):
        # 获取输入 + 标签
        X = batch['X'].squeeze(0).to(device)                # [N_hits, D]
        mask_label = batch['mask_labels'].squeeze(0).to(device)        # [Q, N_hits]
        track_label = batch['track_labels'].squeeze(0).to(device)  # [Q]
        param_label = batch['track_params'].squeeze(0).to(device)  # [Q, 3]

        # Forward pass
        out = model(X)
        # out = {'track_logits': [Q], 'hit_assignment': [Q, N_hits], 'track_properties': [Q, 6]}

        # 只保留前 K 个 hits
        topk_idx = out['topk_idx']
        mask_label = mask_label[:, topk_idx]


        print("track_logits:", out['track_logits'].shape)     # 应该 [Q]
        print("track_labels:", track_label.shape)            # 应该 [Q]
        print("hit_assignment:", out['hit_assignment'].shape) # [Q, K]
        print("mask_labels:", mask_label.shape)              # [Q, K]

        # Compute loss
        loss_dict = loss_fn(
            track_logits=out['track_logits'],
            hit_masks=out['hit_assignment'],
            track_props=out['track_properties'],
            target_cls=track_label,
            target_masks=mask_label,
            target_props=param_label
        )

        # Backward pass
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        total_loss += loss_dict['total'].item()

        # 打印每 10 个 batch 的损失信息
        if i % 10 == 0:
            print(f"Batch {i} - Total Loss: {loss_dict['total']:.4f} | "
                  f"Cls: {loss_dict['cls']:.4f} | "
                  f"Mask: {loss_dict['mask']:.4f} | "
                  f"Param: {loss_dict['param']:.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch Done - Avg Loss: {avg_loss:.4f}")




loss_fn = LossModule()

data_dir = 'data/train_sample/train_100_events'
detectors = pd.read_csv('data/detectors.csv')
dataset = TrackMLDataset(data_dir=data_dir, detectors_df=detectors)

model = UniTrackFormer(input_dim = dataset.feature_dim)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_one_epoch(model, loss_fn, loader, optimizer)

