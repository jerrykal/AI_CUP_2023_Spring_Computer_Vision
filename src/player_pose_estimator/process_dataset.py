import argparse
import os

import cv2
import json_tricks as json
import numpy as np
from mmdet.apis import init_detector
from mmpose.apis import init_model as init_pose_estimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
from predict import predict
from utils import flush_to_csv, flush_to_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
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
        "--device", help="Device to use for inference", type=str, default="cuda:0"
    )
    parser.add_argument(
        "--save_video", help="Save video with estimated poses", action="store_true"
    )

    return parser.parse_args()


def main(args):
    dataset_root = args.dataset_root
    det_config = args.det_config
    det_checkpoint = args.det_checkpoint
    pose_config = args.pose_config
    pose_checkpoint = args.pose_checkpoint
    device = args.device

    # Build detector
    detector = init_detector(det_config, det_checkpoint, device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # Build pose estimator
    pose_estimator = init_pose_estimator(pose_config, pose_checkpoint, device)

    if args.save_video:
        # Build visualizer
        pose_estimator.cfg.visualizer.radius = 3
        pose_estimator.cfg.visualizer.alpha = 1
        pose_estimator.cfg.visualizer.line_width = 1
        visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
        visualizer.set_dataset_meta(
            pose_estimator.dataset_meta, skeleton_style="mmpose"
        )

    for data in sorted(os.listdir(dataset_root)):
        data_root = os.path.join(dataset_root, data)

        video_path = os.path.join(data_root, f"{data}.mp4")
        court_path = os.path.join(data_root, f"{data}_court.csv")

        print(f"Processing {video_path}... ", end="", flush=True)

        try:
            near_player_poses, far_player_poses, all_poses = predict(
                video_path, court_path, detector, pose_estimator
            )
        except Exception as e:
            with open("exceptions.txt", "a") as f:
                f.write(
                    f"Failed to process {video_path} with the following exception:\n {e}\n"
                )

            print("Failed")
            continue

        if args.save_video:
            # Save video
            flush_to_video(
                video_path,
                video_path.replace(".mp4", "_player_poses.mp4"),
                near_player_poses,
                far_player_poses,
                visualizer,
            )

        # Save near and far player poses to csv
        flush_to_csv(
            video_path.replace(".mp4", "_player_poses.csv"),
            near_player_poses,
            far_player_poses,
        )

        # Save full predictions so I don't need to re-run the whole process again
        with open(video_path.replace(".mp4", "_all_poses.json"), "w") as f:
            pred_instances_list = [
                dict(
                    frame_id=i,
                    instances=split_instances(
                        merge_data_samples(all_poses[i]).get("pred_instances", None)
                    ),
                )
                for i in range(len(all_poses))
            ]

            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list,
                ),
                f,
                indent="\t",
            )

        print(f"Done")


if __name__ == "__main__":
    args = parse_args()
    main(args)
