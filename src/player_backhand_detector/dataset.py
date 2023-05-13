import os
import pandas as pd
import yaml
from typing import Dict
import cv2
import ntpath

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BackhandDataset(Dataset):
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


def load_features(hit_labels_path, trajectory_path, player_poses_path, court_path, width=1280, height=720):
    df_hit = pd.read_csv(hit_labels_path)
    df_traj = pd.read_csv(trajectory_path)
    df_pose = pd.read_csv(player_poses_path)
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
    for i in range(0, len(hit_frames)):
        new_feature = np.array(
            [
                [
                    # *court_corners,
                    # df_traj.loc[df_traj["Frame"] == fnum].X.values[0] / width,
                    # df_traj.loc[df_traj["Frame"] == fnum].Y.values[0] / height,
                    *df_pose.loc[fnum].values[1:-1]
                ]
                for fnum in range(max(0, hit_frames[i] - 14), min(hit_frames[i] + 14, len(df_traj) ))
            ]
        )
        new_feature[new_feature != 0] += 1
        features.append(new_feature)

    # last_feature = np.array(
    #     [
    #         [
    #             1 if len(hit_frames) == 1 else 0,
    #             *court_corners,
    #             df_traj.loc[df_traj["Frame"] == fnum].X.values[0] / width,
    #             df_traj.loc[df_traj["Frame"] == fnum].Y.values[0] / height,
    #         ]
    #         for fnum in range(hit_frames[-1], len(df_traj))
    #     ]
    # )
    # last_feature[last_feature != 0] += 1
    # features.append(last_feature)

    return features


def load_labels(labels_path):
    df = pd.read_csv(labels_path)
    return (df["Backhand"].values[:] - 1).tolist()


if __name__ == "__main__":
    ### test
    df_traj = pd.read_csv("datasets/rally/train/00001/00001_trajectory.csv")
    df_pose = pd.read_csv("datasets/rally/train/00001/00001_player_poses.csv")
    df_label = pd.read_csv("datasets/rally/train/00001/00001_S2.csv")
    # print(*df_pose.loc[0].values[1:-1])
    # print(df_traj.loc[df_traj["Frame"] == 1].Y.values[0])
    print(df_label["Backhand"].values[:] - 1)
    print(len(df_traj))
    base_dir = "datasets/rally/train"
    train_features = []
    train_labels = []
    for data_id in list(os.listdir(base_dir)):
        if data_id != "00001":
            break
        features = load_features(
            os.path.join(base_dir, data_id, f"{data_id}_S2.csv"),
            os.path.join(base_dir, data_id, f"{data_id}_trajectory.csv"),
            os.path.join(base_dir, data_id, f"{data_id}_player_poses.csv"),
            os.path.join(base_dir, data_id, f"{data_id}_court.csv"),
        )
        labels = load_labels(
            os.path.join(base_dir, data_id, f"{data_id}_S2.csv")
        )

        train_features += features
        train_labels += labels

    # Create training dataset
    train_dataset = BackhandDataset(train_features, train_labels)
    print(train_dataset.labels.__len__())
    print(train_dataset.features.__len__())
    print(train_dataset.features.shape)
    print(train_dataset.labels.shape)