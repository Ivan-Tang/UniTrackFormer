import numpy as np
from trackml.score import score_event
import torch
from tqdm import tqdm


def compute_efficiency(true_labels, pred_labels):
    """
    计算 efficiency：被正确识别的真实轨迹 hit 数 / 所有真实 hit 数
    """
    true_pids = set(true_labels) - {0} 
    matched_pids = set()

    for pid in true_pids:
        idx = np.where(true_labels == pid)[0]
        pred_ids, counts = np.unique(pred_labels[idx], return_counts=True)

        if len(counts) == 0:
            continue

        max_match = np.max(counts)
        if max_match / len(idx) > 0.5:
            matched_pids.add(pid)

    return len(matched_pids) / len(true_pids) if true_pids else 0.0


def compute_fake_rate(true_labels, pred_labels):
    """
    计算 fake rate：被错误识别为轨迹的 hit 数 / 所有预测轨迹 hit 数
    """
    pred_track_ids = set(pred_labels)
    fake_hits = 0
    total_pred_hits = 0

    for track_id in pred_track_ids:
        if track_id == 0:
            continue
        idx = np.where(pred_labels == track_id)[0]
        total_pred_hits += len(idx)

        true_ids, counts = np.unique(true_labels[idx], return_counts=True)
        main_count = np.max(counts)
        noise_count = len(idx) - main_count
        fake_hits += noise_count

    return fake_hits / total_pred_hits if total_pred_hits > 0 else 0.0


def evaluate_metrics(true_labels, pred_labels):
    """
    综合评估函数，返回多个指标
    """
    efficiency = compute_efficiency(true_labels, pred_labels)
    fake_rate = compute_fake_rate(true_labels, pred_labels)
    return {
        "efficiency": efficiency,
        "fake_rate": fake_rate,
    }


def evaluate_on_test(model, loss_fn, test_loader, model_path, device="cpu"):
    print("Evaluating on test set...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    total_loss = 0
    total_score = 0
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
