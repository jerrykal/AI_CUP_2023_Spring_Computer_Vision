"""
Organize input feature and labels for the hit detection model based on results
from previous annotation pipelines.
"""

import os
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)

    return parser.parse_args()


def organize_x(frame_count, width, height, df_traj, df_court, df_pose):
    court_coords = df_court.values[0]

    row_list = []
    for frame in range(frame_count):
        row_dict = {}
        row_dict["frame"] = frame

        # Court
        (
            row_dict["court_ul_x"],
            row_dict["court_ur_x"],
            row_dict["court_lr_x"],
            row_dict["court_ll_x"],
        ) = (
            court_coords[[0, 2, 4, 6]] / width
        )
        (
            row_dict["court_ul_y"],
            row_dict["court_ur_y"],
            row_dict["court_lr_y"],
            row_dict["court_ll_y"],
        ) = (
            court_coords[[1, 3, 5, 7]] / height
        )

        # Shuttlecock
        shuttle_coords = df_traj.loc[df_traj["Frame"] == frame]
        if not shuttle_coords.empty:
            row_dict["shuttlecock_x"] = shuttle_coords["X"].values[0] / width
            row_dict["shuttlecock_y"] = shuttle_coords["Y"].values[0] / height
        else:
            row_dict["shuttlecock_x"] = 0
            row_dict["shuttlecock_y"] = 0

        # Player poses
        player_poses = df_pose.loc[df_pose["frame"] == frame]
        if not player_poses.empty:
            for i in range(17):
                row_dict[f"near_{i}_x"] = player_poses[f"near_{i}_x"].values[0] / width
                row_dict[f"near_{i}_y"] = player_poses[f"near_{i}_y"].values[0] / height
                row_dict[f"far_{i}_x"] = player_poses[f"far_{i}_x"].values[0] / width
                row_dict[f"far_{i}_y"] = player_poses[f"far_{i}_y"].values[0] / height
        else:
            for i in range(17):
                row_dict[f"near_{i}_x"] = 0
                row_dict[f"near_{i}_y"] = 0
                row_dict[f"far_{i}_x"] = 0
                row_dict[f"far_{i}_y"] = 0

        row_list.append(row_dict)

    df_x = pd.DataFrame(row_list)
    return df_x


def organize_y(frame_count, df_label):
    df_y = pd.DataFrame(columns=["frame", "player_hit"])
    df_y.frame = np.array(range(frame_count))

    # 0: no hit, 1: near player hit, 2: far player hit
    near_hit_frames = df_label.loc[df_label["Hitter"] == "B"]["HitFrame"].values
    far_hit_frames = df_label.loc[df_label["Hitter"] == "A"]["HitFrame"].values
    df_y.player_hit = 0
    df_y.loc[near_hit_frames, "player_hit"] = 1
    df_y.loc[far_hit_frames, "player_hit"] = 2

    return df_y


def main(args):
    data_dir = args.data_dir

    for data in sorted(os.listdir(data_dir)):
        print(f"Processing {data}... ", end="", flush=True)

        # Read results from previous pipelines
        df_traj = pd.read_csv(
            os.path.join(args.data_dir, data, f"{data}_trajectory.csv")
        )
        df_court = pd.read_csv(os.path.join(args.data_dir, data, f"{data}_court.csv"))
        df_pose = pd.read_csv(
            os.path.join(args.data_dir, data, f"{data}_player_poses.csv")
        )

        cap = cv2.VideoCapture(os.path.join(args.data_dir, data, f"{data}.mp4"))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Organize input features
        df_x = organize_x(frame_count, width, height, df_traj, df_court, df_pose)
        df_x.to_csv(
            os.path.join(args.data_dir, data, f"{data}_hitnet_x.csv"), index=False
        )

        # Organize labels
        label_csv = os.path.join(args.data_dir, data, f"{data}_S2.csv")
        if os.path.isfile(label_csv):
            df_label = pd.read_csv(label_csv)
            df_y = organize_y(frame_count, df_label)
            df_y.to_csv(
                os.path.join(args.data_dir, data, f"{data}_hitnet_y.csv"),
                index=False,
            )

        print("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
