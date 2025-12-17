import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TrackMLDataset
from trackformer import TrackFormer
from losses import LossModule
from utils import get_device

config = {
    'epochs': 50,
    'learning_rate': 4e-3,
}

model_config = {
    'd_model': 128,
    'n_heads': 4,
    'hit_filter_layers': 6,
    'track_encoder_layers': 6,
    'track_decoder_layers': 4,
    'n_queries': 100,
    'hit_filter_window': 512,
    'track_window': 256,
    'filter_threshold': 0.0001  # 降低阈值避免过滤掉所有hits
}


def train(model, loss_fn, dataloader, optimizer=None, device='cpu'):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0

    for batch in dataloader:
        X = batch['X'].squeeze(0).to(device)
        mask_label = batch['mask_labels'].squeeze(0).to(device)
        track_label = batch['track_labels'].squeeze(0).to(device)
        param_label = batch['track_params'].squeeze(0).to(device)

        r, phi, z = X[..., 0], X[..., 1], X[..., 2]
        coords = torch.stack([r, phi, z], dim=1)

        out = model(X, coords)
        
        # 检查是否有hits通过过滤
        filtered_indices = out['filtered_indices']  # 布尔掩码 [n_hits]
        n_filtered = filtered_indices.sum().item()
        
        print(f"Debug - filtered_indices shape: {filtered_indices.shape}")
        print(f"Debug - filtered_indices.sum(): {n_filtered}")
        
        # 如果没有hits通过过滤，跳过这个batch
        if n_filtered == 0:
            print("警告: 没有hits通过过滤，跳过这个batch")
            continue
        
        print(f"Debug - mask_label shape: {mask_label.shape}")
        print(f"Debug - X shape: {X.shape}")
        
        # 处理hit filtering - 调整掩码标签以匹配过滤后的hits
        if filtered_indices.size(0) == mask_label.size(1):
            # 维度匹配，应用过滤
            filtered_mask_label = mask_label[:, filtered_indices]
            print(f"Debug - 成功过滤，filtered_mask_label shape: {filtered_mask_label.shape}")
        else:
            # 维度不匹配，这通常发生在数据预处理的hit索引和模型输入的hit不一致
            print(f"警告: 维度不匹配 - filtered_indices {filtered_indices.shape} vs mask_label {mask_label.shape}")
            
            # 重新构建过滤后的mask标签
            # 假设filtered_indices的长度对应输入hits X的数量
            if filtered_indices.size(0) == X.size(0):
                # 对mask_label按照filtered_indices进行过滤
                n_original_hits = mask_label.size(1)
                n_input_hits = X.size(0)
                
                if n_input_hits <= n_original_hits:
                    # 如果输入hits数少于或等于原始hits，取前n_input_hits列
                    truncated_mask_label = mask_label[:, :n_input_hits]
                    filtered_mask_label = truncated_mask_label[:, filtered_indices]
                    print(f"Debug - 截断并过滤后，filtered_mask_label shape: {filtered_mask_label.shape}")
                else:
                    # 如果输入hits数多于原始hits，可能是数据预处理问题
                    print("错误: 输入hits数量大于原始mask_label的hits数量")
                    # 回退：使用原始mask_label，但这可能导致后续维度问题
                    filtered_mask_label = mask_label
            else:
                print("错误: filtered_indices大小与输入X不匹配")
                filtered_mask_label = mask_label

        # 获取维度信息
        n_queries = out['track_logits'].size(0)  # 100 (模型固定查询数)
        n_tracks = track_label.size(0)           # 实际轨道数 (如64)
        
        # 准备真实标签 - 转换为二元分类标签 (0=无轨道, 1=有轨道)
        # 这里我们创建一个简单的标签，实际的匹配由损失函数中的匈牙利算法处理
        binary_track_labels = torch.ones(n_tracks, device=device)  # 所有真实轨道都是正样本
        
        # 验证维度一致性
        n_filtered_hits = out['hit_assignment'].size(1) if len(out['hit_assignment'].shape) > 1 else 0
        expected_mask_hits = filtered_mask_label.size(1) if len(filtered_mask_label.shape) > 1 else 0
        
        print(f"Debug - 模型输出 hit_assignment shape: {out['hit_assignment'].shape}")
        print(f"Debug - 过滤后 mask_label shape: {filtered_mask_label.shape}")
        
        # 确保 hit_assignment 和 filtered_mask_label 的 hits 维度匹配
        if n_filtered_hits != expected_mask_hits and n_filtered_hits > 0:
            print(f"警告: hit维度不匹配 - 模型输出 {n_filtered_hits} vs 标签 {expected_mask_hits}")
            
            # 如果模型输出的hits数少于标签hits数，截断标签
            if n_filtered_hits < expected_mask_hits:
                filtered_mask_label = filtered_mask_label[:, :n_filtered_hits]
                print(f"截断标签到 {filtered_mask_label.shape}")
            # 如果模型输出的hits数多于标签hits数，填充标签
            elif n_filtered_hits > expected_mask_hits:
                padding = torch.zeros(filtered_mask_label.size(0), 
                                    n_filtered_hits - expected_mask_hits, 
                                    device=filtered_mask_label.device)
                filtered_mask_label = torch.cat([filtered_mask_label, padding], dim=1)
                print(f"填充标签到 {filtered_mask_label.shape}")
        
        # 调用损失函数
        loss_dict = loss_fn(
            track_logits = out['track_logits'],          # [100]
            hit_masks = out['hit_assignment'],           # [100, n_filtered_hits]  
            track_props = out['track_properties'],       # [100, 6]
            target_cls = binary_track_labels,            # [64] - 真实轨道标签
            target_masks = filtered_mask_label,          # [64, n_filtered_hits] - 过滤后的掩码
            target_props = param_label,                  # [64, 6] - 轨道参数
            intermediate_outputs = out.get('intermediate_outputs', None)
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
    # 暂时使用CPU避免MPS兼容性问题
    
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

    train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

    train_ids = all_event_ids[:int(N*train_ratio)]
    val_ids = all_event_ids[int(N*train_ratio):int(N*(train_ratio + val_ratio))]
    test_ids = all_event_ids[int(N*(train_ratio + val_ratio)):]

    train_dataset = TrackMLDataset(data_dir, detectors, train_ids)
    val_dataset = TrackMLDataset(data_dir, detectors, val_ids)
    test_dataset = TrackMLDataset(data_dir, detectors, test_ids)

    model = TrackFormer(input_dim = train_dataset.feature_dim, **model_config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'])

    num_epochs = config['epochs']
    train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    # 确保检查点目录存在
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(num_epochs):
        try:
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
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1} completed.")

    plot_loss(train_losses, val_losses, 'checkpoints/loss.png')

    #predict on test
    try:
        checkpoints = torch.load('checkpoints/best_model.pth', map_location = device, weights_only=True)
        model.load_state_dict(checkpoints['model_state_dict'])
        model.eval()
        test_loss = train(model, loss_fn, test_loader, None, device = device)
        print(f"Test Set Loss: {test_loss:.4f}")
    except Exception as e:
        print(f"测试阶段出现错误: {e}")

if __name__ == '__main__':
    main()




