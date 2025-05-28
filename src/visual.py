import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_event_data(event_dir, event_id):
    hits = pd.read_csv(os.path.join(event_dir, f"{event_id}-hits.csv"))
    truth_path = os.path.join(event_dir, f"{event_id}-truth.csv")
    truth = pd.read_csv(truth_path) if os.path.exists(truth_path) else None
    return hits, truth


def visualize_hits_3d(hits, title="3D Hits Distribution"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(hits['x'], hits['y'], hits['z'], s=1, alpha=0.5)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    plt.title(title)
    plt.savefig('results/3d_hits.png')


def visualize_hits_rz(hits, title="Hits in rz Plane"):
    r = np.sqrt(hits['x']**2 + hits['y']**2)
    plt.figure(figsize=(8, 6))
    plt.scatter(hits['z'], r, s=1, alpha=0.3)
    plt.xlabel('Z (mm)')
    plt.ylabel('R = sqrt(x² + y²) (mm)')
    plt.title(title)
    plt.savefig('results/hits_rz.png')


def visualize_ground_truth(hits, truth, title="Ground Truth Trajectories"):
    merged = pd.merge(hits, truth, on='hit_id')
    r = np.sqrt(merged['x']**2 + merged['y']**2)
    
    plt.figure(figsize=(10, 6))
    for pid, group in merged.groupby('particle_id'):
        if pid == 0:
            continue  # Skip noise
        plt.scatter(group['z'], np.sqrt(group['x']**2 + group['y']**2), s=1)
    plt.xlabel('Z (mm)')
    plt.ylabel('R (mm)')
    plt.title(title)
    plt.savefig('results/ground_truth.png')


def visualize_predictions(hits, hit_assignment, threshold=0.5, title="Predicted Tracks"):
    if isinstance(hit_assignment, np.ndarray):
        pred_masks = hit_assignment > threshold
    else:
        pred_masks = (hit_assignment > threshold).cpu().numpy()

    r = np.sqrt(hits['x']**2 + hits['y']**2)

    plt.figure(figsize=(10, 6))
    for qid in range(pred_masks.shape[0]):
        hit_indices = np.where(pred_masks[qid])[0]
        selected_hits = hits.iloc[hit_indices]
        plt.scatter(selected_hits['z'], np.sqrt(selected_hits['x']**2 + selected_hits['y']**2), s=1, label=f'Track {qid}')
    plt.xlabel('Z (mm)')
    plt.ylabel('R (mm)')
    plt.title(title)
    plt.legend(markerscale=5, loc='upper right', bbox_to_anchor=(1.2, 1.0))
    plt.savefig('results/predictions.png')


if __name__ == '__main__':
    # 示例
    event_dir = 'data/train_1_events/'
    event_id = 'event000001000'

    hits, truth = load_event_data(event_dir, event_id)
    visualize_hits_3d(hits)
    visualize_hits_rz(hits)

    if truth is not None:
        visualize_ground_truth(hits, truth)

    # 模拟一个预测输出 (随机掩码演示)
    N_hits = hits.shape[0]
    Q = 5
    fake_assignment = np.random.rand(Q, N_hits)
    visualize_predictions(hits, fake_assignment, threshold=0.9)