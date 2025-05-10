import numpy as np

def compute_efficiency(true_labels, pred_labels):
    """
    计算 efficiency：被正确识别的真实轨迹 hit 数 / 所有真实 hit 数
    """
    true_pids = set(true_labels) - {0}  # 0 表示噪声
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
        'efficiency': efficiency,
        'fake_rate': fake_rate,
    }
