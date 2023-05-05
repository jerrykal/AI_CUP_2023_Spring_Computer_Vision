from mmpose.apis import inference_topdown
from mmpose.evaluation.functional import nms

try:
    from mmdet.apis import inference_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

import numpy as np


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


def detect(
    image,
    player_poses,
    detector,
    pose_estimator,
):
    player_locations = []

    # Predict whole body poses
    pose_preds = predict_all_poses(image, detector, pose_estimator)

    # player_poses[0] should be near player's pose, player_poses[1] should be far player's pose
    for player_pose in player_poses:
        shortest_dist = np.inf
        nearest_pose = None
        for pose_pred in pose_preds:
            dist = np.linalg.norm(
                pose_pred.pred_instances.keypoints[0][16] - player_pose[16]
            )
            if dist < shortest_dist:
                shortest_dist = dist
                nearest_pose = pose_pred

        player_locations.append(nearest_pose.pred_instances.keypoints[0][20])

    return player_locations
