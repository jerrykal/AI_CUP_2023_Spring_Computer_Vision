from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
from utils import flush_to_csv, flush_to_video


def predict_all_poses(img, detector, pose_estimator):
    # Predict bbox
    det_result = inference_detector(detector, img)
    pred_instances = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instances.bboxes, pred_instances.scores[:, None]), axis=1
    )
    bboxes = bboxes[
        np.logical_and(pred_instances.labels == 0, pred_instances.scores > 0.6)
    ]
    bboxes = bboxes[nms(bboxes, 0.3), :4]

    # Predict keypoints
    pose_preds = inference_topdown(pose_estimator, img, bboxes)

    return pose_preds


def predict(
    video_path,
    court_path,
    detector,
    pose_estimator,
):
    # Load video
    cap = cv2.VideoCapture(video_path)

    # Load coordinate of court corners and relax the boundry by 20 pixels
    court_df = pd.read_csv(court_path)
    court_pts = np.array(
        [
            [
                [court_df.upper_left_x[0] - 20, court_df.upper_left_y[0] - 20],
                [court_df.upper_right_x[0] + 20, court_df.upper_right_y[0] - 20],
                [court_df.lower_right_x[0] + 20, court_df.lower_right_y[0] + 20],
                [court_df.lower_left_x[0] - 20, court_df.lower_left_y[0] + 20],
            ]
        ]
    )

    all_poses = []
    near_player_poses = []
    far_player_poses = []

    # Process video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        player_pose_candidates = []

        # Estimate all poses presented in the video frame
        pose_preds = predict_all_poses(frame, detector, pose_estimator)
        all_poses.append(pose_preds)

        if len(pose_preds) < 2:
            near_player_poses.append(None)
            far_player_poses.append(None)
            continue

        dist_to_last_near_player = [np.inf] * len(pose_preds)
        dist_to_last_far_player = [np.inf] * len(pose_preds)

        # Filter out poses with its left foot outside the court
        for idx, pose in enumerate(pose_preds):
            pred_instances = pose.get("pred_instances", None)
            if pred_instances is None:
                continue

            keypoints = pred_instances.keypoints[0]
            if len(keypoints) < 17:
                continue

            left_foot = (int(keypoints[15][0]), int(keypoints[15][1]))

            if (
                len(player_pose_candidates) < 2
                and cv2.pointPolygonTest(court_pts, left_foot, False) != -1
            ):
                player_pose_candidates.append(idx)
                continue

            if len(near_player_poses) > 0 and near_player_poses[-1] is not None:
                dist_to_last_near_player[idx] = np.linalg.norm(
                    np.array(near_player_poses[-1].pred_instances.keypoints[0][15])
                    - np.array(left_foot)
                )

            if len(far_player_poses) > 0 and far_player_poses[-1] is not None:
                dist_to_last_far_player[idx] = np.linalg.norm(
                    np.array(far_player_poses[-1].pred_instances.keypoints[0][15])
                    - np.array(left_foot)
                )

        if len(player_pose_candidates) == 2:
            left_foot_y = [
                pose_preds[i].pred_instances.keypoints[0][15][1]
                for i in player_pose_candidates
            ]
            near_player_poses.append(
                pose_preds[player_pose_candidates[np.argmax(left_foot_y)]]
            )
            far_player_poses.append(
                pose_preds[player_pose_candidates[np.argmin(left_foot_y)]]
            )
        elif (
            len(near_player_poses) > 1
            and len(far_player_poses) > 1
            and near_player_poses[-1] is not None
            and far_player_poses[-1] is not None
        ):
            if len(player_pose_candidates) == 1:
                candidate_pose = pose_preds[player_pose_candidates[0]]
                candidate_left_foot = candidate_pose.pred_instances.keypoints[0][15]

                dist_1 = np.linalg.norm(
                    np.array(near_player_poses[-1].pred_instances.keypoints[0][15])
                    - np.array(candidate_left_foot)
                )
                dist_2 = np.linalg.norm(
                    np.array(far_player_poses[-1].pred_instances.keypoints[0][15])
                    - np.array(candidate_left_foot)
                )

                if dist_1 < dist_2:
                    near_player_poses.append(candidate_pose)

                    far_cand = np.argmin(dist_to_last_far_player)
                    if dist_to_last_far_player[far_cand] < 30:
                        far_player_poses.append(pose_preds[far_cand])
                    else:
                        far_player_poses.append(far_player_poses[-1])
                else:
                    near_cand = np.argmin(dist_to_last_near_player)
                    if dist_to_last_near_player[near_cand] < 30:
                        near_player_poses.append(pose_preds[near_cand])
                    else:
                        near_player_poses.append(near_player_poses[-1])

                    far_player_poses.append(candidate_pose)
            else:
                near_cand = np.argmin(dist_to_last_near_player)
                if dist_to_last_near_player[near_cand] < 30:
                    near_player_poses.append(pose_preds[near_cand])
                else:
                    near_player_poses.append(near_player_poses[-1])

                far_cand = np.argmin(dist_to_last_far_player)
                if dist_to_last_far_player[far_cand] < 30:
                    far_player_poses.append(pose_preds[far_cand])
                else:
                    far_player_poses.append(far_player_poses[-1])
        else:
            near_player_poses.append(
                None if len(near_player_poses) == 0 else near_player_poses[-1]
            )
            far_player_poses.append(
                None if len(far_player_poses) == 0 else far_player_poses[-1]
            )

    cap.release()

    return near_player_poses, far_player_poses, all_poses


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--video_path", help="Input video file", type=str, required=True
    )
    parser.add_argument(
        "--court_path",
        help="File containing coordinates of badminton court corners",
        type=str,
        required=True,
    )
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

    # Parse command line arguments
    args = parser.parse_args()

    # Build detector
    detector = init_detector(args.det_config, args.det_checkpoint, args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # Build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config, args.pose_checkpoint, args.device
    )

    # Estimate player poses
    near_player_poses, far_player_poses, _ = predict(
        args.video_path, args.court_path, detector, pose_estimator
    )

    if args.save_video:
        # Build visualizer
        pose_estimator.cfg.visualizer.radius = 3
        pose_estimator.cfg.visualizer.alpha = 1
        pose_estimator.cfg.visualizer.line_width = 1
        visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
        visualizer.set_dataset_meta(
            pose_estimator.dataset_meta, skeleton_style="mmpose"
        )

        # Save video
        flush_to_video(
            args.video_path,
            args.video_path.replace(".mp4", "_player_pose.mp4"),
            near_player_poses,
            far_player_poses,
            visualizer,
        )

    # Save near and far player poses to csv
    flush_to_csv(
        args.video_path.replace(".mp4", "_player_pose.csv"),
        near_player_poses,
        far_player_poses,
    )


if __name__ == "__main__":
    main()
