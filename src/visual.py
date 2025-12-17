import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from dataset import TrackMLDataset
from trackformer import create_trackformer_600mev
from trackml.dataset import load_event
from utils import get_device


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
    print('3d_hits.png saved to results/')


def visualize_hits_rz(hits, title="Hits in rz Plane"):
    r = np.sqrt(hits['x']**2 + hits['y']**2)
    plt.figure(figsize=(8, 6))
    plt.scatter(hits['z'], r, s=1, alpha=0.3)
    plt.xlabel('Z (mm)')
    plt.ylabel('R = sqrt(x² + y²) (mm)')
    plt.title(title)
    plt.savefig('results/hits_rz.png')
    print('hits_rz.png saved to results/')
    


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
    print('ground_truth.png saved to results/')


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
    print('predictions.png saved to results/')


if __name__ == '__main__':
    event_dir = 'data/sample_event/'
    event_id = 'event000001000'

    print('Plotting hits...')
    hits, truth = load_event_data(event_dir, event_id)
    visualize_hits_3d(hits)
    visualize_hits_rz(hits)

    print('Plotting ground truth...')
    if truth is not None:
        visualize_ground_truth(hits, truth)

    print('Making Predictions...')
    #load model and predict
    device = get_device()
    detector_df = pd.read_csv('data/detectors.csv')
    hits, cells, particles, truth = load_event(os.path.join(event_dir, event_id))

    X = TrackMLDataset.extract_features(hits, cells, detector_df)
    print(X)
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)

    model = create_trackformer_600mev(input_dim=X.shape[1])
    checkpoints = torch.load('checkpoints/best_model.pth', weights_only=True)
    model.load_state_dict(checkpoints['model_state_dict'])
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        prediction = model(X_tensor)
        print(visualize_predictions)
        hit_assignment = torch.sigmoid(prediction['hit_assignment']).cpu().numpy()
        print(hit_assignment)
 
    visualize_predictions(hits, hit_assignment, threshold=0.5)