import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import TrackMLDataset
from src.models import UniTrackFormer
from src.losses import LossModule

def train(model, loss_fn, dataloader, optimizer=None, device='cpu'):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0

    for batch in dataloader:
        X = batch['X'].squeeze(0).to(device)
        mask_label = batch['mask_labels'].squeeze(0).to(device)
        track_label = batch['track_labels'].squeeze(0).to(device)
        param_label = batch['track_params'].squeeze(0).to(device)

        out = model(X)
        topk_idx = out['topk_idx'].cpu().numpy()
        mask_label = mask_label[:, topk_idx]

        loss_dict = loss_fn(
            track_logits = out['track_logits'],
            hit_masks = out['hit_assignment'],
            track_props = out['track_properties'],
            target_cls = track_label,
            target_masks = mask_label,
            target_props = param_label,
        )

        if is_train:
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()

        total_loss += loss_dict['total'].item()

    avg_loss = total_loss / len(dataloader)

    return avg_loss

def plot_loss(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label = 'Train')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label = 'Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    def get_device():
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
        
    device = get_device()
    print(f'Training on {device}')
    loss_fn = LossModule()

    data_dir = 'data/train_sample'
    detectors = pd.read_csv('data/detectors.csv')

    all_event_ids = sorted(
        set(
            fname.split('-')[0]
            for fname in os.listdir(data_dir) if fname.endswith('-hits.csv')
        )
    )
    N = len(all_event_ids)

    train_ratio, val_ratio, test_ration = 0.7, 0.2, 0.1

    train_ids = all_event_ids[:int(N*train_ratio)]
    val_ids = all_event_ids[int(N*train_ratio):int(N*(train_ratio + val_ratio))]
    test_ids = all_event_ids[int(N*(train_ratio + val_ratio)):]

    train_dataset = TrackMLDataset(data_dir, detectors, train_ids)
    val_dataset = TrackMLDataset(data_dir, detectors, val_ids)
    test_dataset = TrackMLDataset(data_dir, detectors, test_ids)

    model = UniTrackFormer(
        input_dim = train_dataset.feature_dim
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    num_epochs = 100
    train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        train_loss = train(model, loss_fn, train_loader, optimizer, device)
        val_loss = train(model, loss_fn, val_loader, device = device)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                },
                'checkpoints/best_model.pth',
            )

    plot_loss(train_losses, val_losses, 'checkpoints/loss.png')


    #predict on test
    checkpoints = torch.load('checkpoints/best_model.pth', map_location = device, weights_only=True)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.eval()
    test_loss = train(model, loss_fn, test_loader, None, device = device)
    print(f"Test Set Loss: {test_loss:.4f}")

if __name__ == '__main__':
    main()



    
