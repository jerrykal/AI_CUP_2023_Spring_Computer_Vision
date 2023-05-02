"""
Custom dataset and dataloader for HitNet.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class HitNetDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

    def __len__(self):
        return len(self.features)

    def get_sample_weights(self):
        """Get sample weights for WeightedRandomSampler."""
        class_count = [sum(self.labels == i) for i in range(3)]
        return [1.0 / class_count[i] for i in self.labels.numpy()]


FEAT_COLS = [
    "court_ul_x",
    "court_ul_y",
    "court_ur_x",
    "court_ur_y",
    "court_lr_x",
    "court_lr_y",
    "court_ll_x",
    "court_ll_y",
    "shuttlecock_x",
    "shuttlecock_y",
]
for player in ["near", "far"]:
    for i in range(17):
        FEAT_COLS.append(f"{player}_{i}_x")
        FEAT_COLS.append(f"{player}_{i}_y")


def load_features(features_path):
    df_x = pd.read_csv(features_path)
    features = df_x[FEAT_COLS].to_numpy()

    # Add one to all the non-zero entries so that detected coordinates are within
    # the range of [1, 2], and undetected coordinates remain zero.
    features[features != 0] += 1
    return features


def load_labels(labels_path):
    df_y = pd.read_csv(labels_path)
    labels = df_y["player_hit"].to_numpy()
    return labels


def concat_frames(features, labels, concat_n, stepsize):
    concat_windows = (
        np.expand_dims(np.arange(0, concat_n), 0)
        + np.expand_dims(np.arange(0, features.shape[0] - concat_n + 1), 0).T
    )

    new_features = features[concat_windows[::stepsize]]
    if labels is None:
        return new_features

    # Label as hit if the second half of the concatenated frames contains a hit.
    new_labels = [
        1
        if 1 in concated_labels[concat_n // 2 :]
        else 2
        if 2 in concated_labels[concat_n // 2 :]
        else 0
        for concated_labels in labels[concat_windows[::stepsize]]
    ]

    return new_features, new_labels


def preprocess_data(dataset_path, data_id, concat_n, stepsize, data_type="train"):
    features = load_features(
        os.path.join(dataset_path, data_id, f"{data_id}_hitnet_x.csv")
    )
    if data_type != "test":
        labels = load_labels(
            os.path.join(dataset_path, data_id, f"{data_id}_hitnet_y.csv")
        )
        features, labels = concat_frames(features, labels, concat_n, stepsize)
        return features, labels

    features = concat_frames(features, None, concat_n, stepsize)
    return features
