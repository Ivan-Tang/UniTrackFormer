import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from trackml.dataset import load_dataset, load_event


class TrackMLDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        detectors_df: pd.DataFrame,
        event_ids: list,
        Q=64,
    ):
        self.data_dir = data_dir
        self.Q = Q
        self.detectors_df = detectors_df
        self.event_ids = event_ids
        self.feature_dim = 27

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx):
        event_id = self.event_ids[idx]

        hits, cells, particles, truth = load_event(
            os.path.join(self.data_dir, f"{event_id}")
        )

        X = self.extract_features(hits, cells, self.detectors_df)  # [N_hits, D]
        self.feature_dim = X.shape[1]

        mask_labels, track_labels, track_params = self.build_labels(
            particles, truth, Q=self.Q
        )

        return {
            "X": torch.tensor(X.values, dtype=torch.float32),
            "mask_labels": torch.tensor(mask_labels, dtype=torch.float32),
            "track_labels": torch.tensor(track_labels, dtype=torch.float32),
            "track_params": torch.tensor(track_params, dtype=torch.float32),
        }

    @staticmethod
    def extract_features(
        hits_df: pd.DataFrame, cells_df: pd.DataFrame, detectors_df: pd.DataFrame
    ):
        """
        计算所有特征并合并
        输出 shape: [N_hits, 16]
        """

        x = hits_df["x"].values
        y = hits_df["y"].values
        z = hits_df["z"].values

        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        hits_features = pd.DataFrame(
            np.stack(
                [
                    hits_df["hit_id"],
                    x,
                    y,
                    z,
                    r,
                    phi,
                    hits_df["volume_id"],
                    hits_df["layer_id"],
                    hits_df["module_id"],
                ],
                axis=1,
            ),
            columns=[
                "hit_id",
                "x",
                "y",
                "z",
                "r",
                "phi",
                "volume_id",
                "layer_id",
                "module_id",
            ],
        )
        grouped = cells_df.groupby("hit_id")

        charge_sum = grouped["value"].sum()
        charge_max = grouped["value"].max()
        n_cells = grouped.size()

        ch0_center = grouped.apply(
            lambda x: np.average(x["ch0"], weights=x["value"]), include_groups=False
        )
        ch1_center = grouped.apply(
            lambda x: np.average(x["ch1"], weights=x["value"]), include_groups=False
        )

        cells_features = pd.DataFrame(
            {
                "hits_id": charge_sum.index,
                "charge_sum": charge_sum.values,
                "charge_max": charge_max.values,
                "n_cells": n_cells,
                "ch0_center": ch0_center,
                "ch1_center": ch1_center,
            }
        )

        detectors_features = detectors_df[
            [
                "volume_id",
                "layer_id",
                "module_id",
                "cx",
                "cy",
                "cz",
                "module_t",
                "module_minhu",
                "module_maxhu",
                "module_hv",
                "pitch_u",
                "pitch_v",
            ]
        ]

        features = pd.merge(hits_features, cells_features, on="hit_id", how="left")
        features = pd.merge(
            features,
            detectors_features,
            on=["volume_id", "layer_id", "module_id"],
            how="left",
        )

        cx = features["cx"].values
        cy = features["cy"].values
        cz = features["cz"].values


        x = features["x"].values
        y = features["y"].values
        z = features["z"].values

        features["dist_to_center"] = np.sqrt(
            (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
        )
        features["module_area"] = (
            1
            / 2
            * (features["module_minhu"] + features["module_maxhu"])
            * features["module_hv"]
        )
        features["cell_area"] = features["pitch_u"] * features["pitch_v"]

        return features

    @staticmethod
    def build_labels(particles_df: pd.DataFrame, truth_df: pd.DataFrame, Q: int = 64):
        """
        输入：
        particles_df: [N_particles, N_params] 所有particles的物理参数:['particle_id', 'vx', 'vy', 'vz', 'px', 'py', 'pz', 'q', 'nhits']
        truth_df: [N_hits, 9] 所有hits对应的particle_id, tx/y/z, tpx/y/z, weight
        Q:单次处理的轨迹数量,默认为64
        返回标签:
        mask labels: [Q, N_hits] 哪些hits属于哪些轨迹
        track labels: [Q] 哪些query是有效轨迹
        track params: [Q, N_params] 轨迹的物理参数
        """

        # 预处理
        particles_df = particles_df[particles_df["nhits"] >= 3].copy()
        truth_df = truth_df[truth_df["particle_id"] != 0].copy()

        # 选择轨迹
        particle_hits = truth_df.groupby("particle_id")["hit_id"].apply(set).to_dict()
        selected_hits = set()
        pid_to_qid = {}
        qid = 0

        # 优先选择nhits多的
        sorted_particles_df = particles_df.sort_values(by="nhits", ascending=False)

        for pid in sorted_particles_df["particle_id"]:
            if pid in particle_hits:
                hits = particle_hits[pid]
                # 如果hits和已有hits不相同，则选择
                if hits not in selected_hits:
                    pid_to_qid[pid] = qid
                    selected_hits.update(hits)
                    qid += 1
                if qid >= Q:
                    break

        N_hits = truth_df["hit_id"].nunique()

        # 创建映射关系
        hit_id_to_idx = {hid: idx for idx, hid in enumerate(truth_df["hit_id"])}

        # 初始化标签矩阵
        mask_labels = np.zeros((Q, N_hits), dtype=np.float32)
        track_labels = np.zeros(Q, dtype=np.float32)
        track_params = np.zeros((Q, 6), dtype=np.float32)

        # 填入标签
        for _, row in truth_df.iterrows():
            hid = row["hit_id"]
            hid_idx = hit_id_to_idx[hid]
            pid = row["particle_id"]
            qid = pid_to_qid.get(pid, -1)
            if qid == -1:
                continue
            mask_labels[qid, hid_idx] = 1.0

        for pid, qid in pid_to_qid.items():
            track_labels[qid] = 1.0 
            row = particles_df[particles_df["particle_id"] == pid].iloc[0]
            track_params[qid] = [
                row["vx"],
                row["vy"],
                row["vz"],
                row["px"],
                row["py"],
                row["pz"],
            ]

        return mask_labels, track_labels, track_params
