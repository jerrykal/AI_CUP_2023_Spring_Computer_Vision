import cv2
import pandas as pd
from mmpose.structures import merge_data_samples


def flush_to_video(
    video_path, output_path, near_player_poses, far_player_poses, visualizer
):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(len(near_player_poses)):
        ret, frame = cap.read()
        if not ret:
            break

        near_player_pose = near_player_poses[i]
        far_player_pose = far_player_poses[i]

        player_poses = []
        if near_player_pose is not None:
            player_poses.append(near_player_pose)
        if far_player_pose is not None:
            player_poses.append(far_player_pose)

        visualizer.add_datasample(
            "result",
            frame,
            data_sample=merge_data_samples(player_poses),
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=False,
            show_kpt_idx=False,
            skeleton_style="mmpose",
            show=False,
            wait_time=0,
            kpt_thr=0.3,
        )

        out.write(visualizer.get_image())

    cap.release()
    out.release()


def flush_to_csv(output_path, near_player_poses, far_player_poses):
    pose_list = []
    for frame in range(len(near_player_poses)):
        pose_dict = {}
        pose_dict["frame"] = frame
        for i in range(17):
            if near_player_poses[frame] is not None:
                near_player_kpt = near_player_poses[frame].pred_instances.keypoints[0][
                    i
                ]
                pose_dict[f"near_{i}_x"] = near_player_kpt[0]
                pose_dict[f"near_{i}_y"] = near_player_kpt[1]
            else:
                pose_dict[f"near_{i}_x"] = 0
                pose_dict[f"near_{i}_y"] = 0

            if far_player_poses[frame] is not None:
                far_player_kpt = far_player_poses[frame].pred_instances.keypoints[0][i]

                pose_dict[f"far_{i}_x"] = far_player_kpt[0]
                pose_dict[f"far_{i}_y"] = far_player_kpt[1]
            else:
                pose_dict[f"far_{i}_x"] = 0
                pose_dict[f"far_{i}_y"] = 0

        pose_list.append(pose_dict)

    pose_df = pd.DataFrame(pose_list)
    pose_df.to_csv(output_path, index=False)
