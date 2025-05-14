import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import TrackMLDataset
from src.losses import LossModule
from src.models import UniTrackFormer


def train_one_epoch(model, loss_fn, dataloader, optimizer, device="cpu"):
    model.train()
    total_loss = 0

    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    device = get_device()
    print(f"Training on {device}...")
    model = model.to(device)

    for i, batch in tqdm(enumerate(dataloader)):
        # 获取输入 + 标签
        X = batch["X"].squeeze(0).to(device)  # [N_hits, D]
        mask_label = batch["mask_labels"].squeeze(0).to(device)  # [Q, N_hits]
        track_label = batch["track_labels"].squeeze(0).to(device)  # [Q]
        param_label = batch["track_params"].squeeze(0).to(device)  # [Q, 3]

        # Forward pass
        out = model(X)
        # out = {'track_logits': [Q], 'hit_assignment': [Q, N_hits], 'track_properties': [Q, 6]}

        # 只保留前 K 个 hits
        topk_idx = out["topk_idx"]
        mask_label = mask_label[:, topk_idx]

        # Compute loss
        loss_dict = loss_fn(
            track_logits=out["track_logits"],
            hit_masks=out["hit_assignment"],
            track_props=out["track_properties"],
            target_cls=track_label,
            target_masks=mask_label,
            target_props=param_label,
        )

        # Backward pass
        optimizer.zero_grad()
        loss_dict["total"].backward()
        optimizer.step()

        total_loss += loss_dict["total"].item()

        if i % 10 == 0:
            print(
                f"Batch {i} - Total Loss: {loss_dict['total']:.4f} | "
                f"Cls: {loss_dict['cls']:.4f} | "
                f"Mask: {loss_dict['mask']:.4f} | "
                f"Param: {loss_dict['param']:.4f}"
            )

        if i % 100 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "batch": i,
                },
                "checkpoints/unitrackformer_checkpoint.pth",
            )

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch Done - Avg Loss: {avg_loss:.4f}")

def validate_one_epoch(model, loss_fn, dataloader, device="cpu"):
    model.eval()
    total_loss = 0

    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    device = get_device()
    model = model.to(device)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            X = batch["X"].squeeze(0).to(device)
            mask_label = batch["mask_labels"].squeeze(0).to(device)
            track_label = batch["track_labels"].squeeze(0).to(device)
            param_label = batch["track_params"].squeeze(0).to(device)

            out = model(X)
            topk_idx = out["topk_idx"]
            mask_label = mask_label[:, topk_idx]

            loss_dict = loss_fn(
                track_logits=out["track_logits"],
                hit_masks=out["hit_assignment"],
                track_props=out["track_properties"],
                target_cls=track_label,
                target_masks=mask_label,
                target_props=param_label,
            )

            total_loss += loss_dict["total"].item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


if __name__ == "__main__":
    import os

    loss_fn = LossModule()

    data_dir = "data/train_10_events"
    detectors = pd.read_csv("data/detectors.csv")

    all_event_ids = sorted(
        set(
            fname.split("-")[0]
            for fname in os.listdir(data_dir)
            if fname.endswith("-hits.csv")
        )
    )
    N = len(all_event_ids)
    train_val_split = 0.8

    train_ids = all_event_ids[: int(train_val_split * N)]
    val_ids = all_event_ids[int(train_val_split * N) :]

    train_dataset = TrackMLDataset(
        data_dir=data_dir, detectors_df=detectors, event_ids=train_ids
    )
    val_dataset = TrackMLDataset(
        data_dir=data_dir, detectors_df=detectors, event_ids=val_ids
    )

    model = UniTrackFormer(input_dim=train_dataset.feature_dim)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loss = train_one_epoch(model, loss_fn, train_loader, optimizer)
    val_loss = validate_one_epoch(model, loss_fn, val_loader)
