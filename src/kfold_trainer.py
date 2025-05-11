import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.dataset import TrackMLDataset
from src.losses import LossModule
from src.models import UniTrackFormer
from src.trainer import train


def train_kfold(
    model_class,
    loss_fn_class,
    dataset_class,
    data_dir,
    detectors,
    event_ids,
    num_folds=5,
    num_epochs=10,
    device="cpu",
):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    all_event_ids = sorted(event_ids)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_event_ids)):
        print(f"Starting Fold {fold + 1}/{num_folds}")

        # 获取训练和验证集的事件 ID
        train_ids = [all_event_ids[i] for i in train_idx]
        val_ids = [all_event_ids[i] for i in val_idx]

        # 创建数据集和数据加载器
        train_dataset = dataset_class(
            data_dir=data_dir, detectors_df=detectors, event_ids=train_ids
        )
        val_dataset = dataset_class(
            data_dir=data_dir, detectors_df=detectors, event_ids=val_ids
        )

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # 初始化模型、损失函数和优化器
        model = model_class(input_dim=train_dataset.feature_dim).to(device)
        loss_fn = loss_fn_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # 训练模型
        train(
            model,
            loss_fn,
            train_loader,
            val_loader,
            optimizer,
            num_epochs=num_epochs,
            device=device,
        )

        # 保存每个折的模型
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "fold": fold,
            },
            f"checkpoints/model_fold_{fold + 1}.pth",
        )

        print(f"Fold {fold + 1} completed.\n")


if __name__ == "__main__":

    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    device = get_device()
    print("Training on device")

    data_dir = "data/train_sample/train_100_events"
    detectors = pd.read_csv("data/detectors.csv")
    all_event_ids = sorted(
        set(
            fname.split("-")[0]
            for fname in os.listdir(data_dir)
            if fname.endswith("-hits.csv")
        )
    )

    train_kfold(
        model_class=UniTrackFormer,
        loss_fn_class=LossModule,
        dataset_class=TrackMLDataset,
        data_dir=data_dir,
        detectors=detectors,
        event_ids=all_event_ids,
        num_folds=5,
        num_epochs=10,
        device=device,
    )
