import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import TrackMLDataset
from src.losses import LossModule
from src.models import UniTrackFormer


def train(
    model, loss_fn, train_loader, val_loader, optimizer, num_epochs=10, device="cpu"
):
    print(f"Training on {device}")
    best_val_loss = float("inf")

    for epoch in tqdm(range(num_epochs)):
        train_loss = train_one_epoch(
            model, loss_fn, train_loader, optimizer, device=device
        )
        val_loss = validate_one_epoch(model, loss_fn, val_loader, device=device)
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                f"checkpoints/best_model.pth",
            )


def train_one_epoch(model, loss_fn, dataloader, optimizer, device="cpu"):
    model.train()
    total_loss = 0
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

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate_one_epoch(model, loss_fn, dataloader, device="cpu"):
    model.eval()
    total_loss = 0
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
    return avg_loss


def evaluate_on_test(model, loss_fn, test_loader, model_path, device="cpu"):
    print("Evaluating on test set...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
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

    avg_loss = total_loss / len(test_loader)
    return avg_loss


if __name__ == "__main__":

    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    device = get_device()
    loss_fn = LossModule()

    data_dir = "data/train_sample/train_10_events"
    detectors = pd.read_csv("data/detectors.csv")

    all_event_ids = sorted(
        set(
            fname.split("-")[0]
            for fname in os.listdir(data_dir)
            if fname.endswith("-hits.csv")
        )
    )
    N = len(all_event_ids)
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    train_ids = all_event_ids[: int(train_ratio * N)]
    val_ids = all_event_ids[int(train_ratio * N) : int((train_ratio + val_ratio) * N)]
    test_ids = all_event_ids[int((train_ratio + val_ratio) * N) :]

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

    train(
        model,
        loss_fn,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=10,
        device=device,
    )

    test_dataset = TrackMLDataset(
        data_dir=data_dir, detectors_df=detectors, event_ids=test_ids
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_loss = evaluate_on_test(
        model,
        loss_fn,
        test_loader,
        model_path="checkpoints/best_model.pth",
        device=device,
    )
    print(f"Test Set Loss: {test_loss:.4f}")
