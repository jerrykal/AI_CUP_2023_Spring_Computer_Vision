import os
from argparse import ArgumentParser

import cv2
import pandas as pd
from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline

from src.player_location_detector.detect import detect

try:
    from mmdet.apis import init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def main(args):
    df_answer = pd.DataFrame(
        columns=[
            "VideoName",
            "ShotSeq",
            "HitFrame",
            "Hitter",
            "RoundHead",
            "Backhand",
            "BallHeight",
            "LandingX",
            "LandingY",
            "HitterLocationX",
            "HitterLocationY",
            "DefenderLocationX",
            "DefenderLocationY",
            "BallType",
            "Winner",
        ]
    )

    # Build detector
    detector = init_detector(args.det_config, args.det_checkpoint, args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # Build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config, args.pose_checkpoint, args.device
    )

    for data_id in sorted(os.listdir(args.dataset_path)):
        print(
            f"Organizing {os.path.join(args.dataset_path, data_id)}... ",
            end="",
            flush=True,
        )

        cap = cv2.VideoCapture(
            os.path.join(args.dataset_path, data_id, f"{data_id}.mp4")
        )

        data_path = os.path.join(args.dataset_path, data_id)

        df_hit = pd.read_csv(os.path.join(data_path, f"{data_id}_hit.csv"))
        df_pose = pd.read_csv(os.path.join(data_path, f"{data_id}_player_poses.csv"))
        df_shot_type = pd.read_csv(os.path.join(data_path, f"{data_id}_shot_type.csv"))
        for _, row in df_hit.iterrows():
            # Extract hit frame image
            cap.set(cv2.CAP_PROP_POS_FRAMES, row["HitFrame"])
            image = cap.read()[1]
            player_poses = [
                [
                    (
                        df_pose[df_pose["frame"] == row["HitFrame"]][
                            f"near_{i}_x"
                        ].values[0],
                        df_pose[df_pose["frame"] == row["HitFrame"]][
                            f"near_{i}_y"
                        ].values[0],
                    )
                    for i in range(17)
                ],
                [
                    (
                        df_pose[df_pose["frame"] == row["HitFrame"]][
                            f"far_{i}_x"
                        ].values[0],
                        df_pose[df_pose["frame"] == row["HitFrame"]][
                            f"far_{i}_y"
                        ].values[0],
                    )
                    for i in range(17)
                ],
            ]

            # Detect player locations
            player_locations = detect(
                image,
                player_poses,
                detector,
                pose_estimator,
            )

            new_row = {
                "VideoName": data_id + ".mp4",
                "ShotSeq": row["ShotSeq"],
                "HitFrame": row["HitFrame"],
                "Hitter": row["Hitter"],
                "RoundHead": 0,
                "Backhand": 0,
                "BallHeight": 1
                if (
                    player_poses[1 if row["Hitter"] == "A" else 0][6][1]
                    - player_poses[1 if row["Hitter"] == "A" else 0][10][1]
                    > 20
                )
                else 2,
                "LandingX": 0,
                "LandingY": 0,
                "HitterLocationX": int(
                    player_locations[1 if row["Hitter"] == "A" else 0][0]
                ),
                "HitterLocationY": int(
                    player_locations[1 if row["Hitter"] == "A" else 0][1]
                ),
                "DefenderLocationX": int(
                    player_locations[0 if row["Hitter"] == "A" else 1][0]
                ),
                "DefenderLocationY": int(
                    player_locations[0 if row["Hitter"] == "A" else 1][1]
                ),
                "BallType": df_shot_type[df_shot_type["ShotSeq"] == row["ShotSeq"]][
                    "BallType"
                ].values[0],
                "Winner": "X",
            }
            df_answer.loc[len(df_answer)] = new_row

        df_answer.Winner.values[-1] = "A" if df_answer.Hitter.values[-1] == "B" else "B"
        cap.release()

        print("Done!")

    df_answer.to_csv("answer.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument(
        "--det_config", help="Config file for detection", type=str, required=True
    )
    parser.add_argument(
        "--det_checkpoint",
        help="Checkpoint file for detection",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pose_config", help="Config file for pose estimation", type=str, required=True
    )
    parser.add_argument(
        "--pose_checkpoint",
        help="Checkpoint file for pose estimation",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device", help="Device for detection", type=str, default="cuda:0"
    )

    main(parser.parse_args())
