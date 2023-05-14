import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BallHeightDataset(Dataset):
    def __init__(self, features, labels=None, num_classes=2):
        self.num_classes = num_classes

        self.features = []
        for feature in features:
            self.features.append(torch.FloatTensor(feature))
        self.features = nn.utils.rnn.pad_sequence(
            self.features, batch_first=True, padding_value=-1
        )

        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

    def __len__(self):
        return len(self.features)

    def get_sample_weights(self):
        """Get sample weights for WeightedRandomSampler."""
        class_count = [sum(self.labels == i) for i in range(0, self.num_classes)]
        print(class_count)
        return [1.0 / class_count[i] for i in self.labels.numpy()]


def load_features(hit_labels_path, trajectory_path, court_path, width=1280, height=720):
    df_hit = pd.read_csv(hit_labels_path)
    df_traj = pd.read_csv(trajectory_path)
    df_court = pd.read_csv(court_path)

    COURT_COLS = [
        "upper_left_x",
        "upper_left_y",
        "upper_right_x",
        "upper_right_y",
        "lower_right_x",
        "lower_right_y",
        "lower_left_x",
        "lower_left_y",
    ]
    court_corners = df_court[COURT_COLS].values[0].astype(np.float32)
    court_corners[::2] /= width
    court_corners[1::2] /= height

    features = []
    hit_frames = df_hit["HitFrame"].values
    for i in range(1, len(hit_frames)):
        new_feature = np.array(
            [
                [
                    1 if i == 1 else 0,
                    *court_corners,
                    df_traj.loc[df_traj["Frame"] == fnum].X.values[0] / width,
                    df_traj.loc[df_traj["Frame"] == fnum].Y.values[0] / height,
                ]
                for fnum in range(hit_frames[i - 1], hit_frames[i])
            ]
        )
        new_feature[new_feature != 0] += 1
        features.append(new_feature)

    last_feature = np.array(
        [
            [
                1 if len(hit_frames) == 1 else 0,
                *court_corners,
                df_traj.loc[df_traj["Frame"] == fnum].X.values[0] / width,
                df_traj.loc[df_traj["Frame"] == fnum].Y.values[0] / height,
            ]
            for fnum in range(hit_frames[-1], len(df_traj))
        ]
    )
    last_feature[last_feature != 0] += 1
    features.append(last_feature)

    return features


def load_labels(labels_path):
    df = pd.read_csv(labels_path)
    return (df["BallHeight"].values[:] - 1).tolist()
