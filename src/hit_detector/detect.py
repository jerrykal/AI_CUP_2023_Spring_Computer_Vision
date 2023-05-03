import math
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataset import HitNetDataset, preprocess_data

from model import HitNet


def inference(model, dataset):
    hit_probs = np.empty((len(dataset), 3), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i, feature in enumerate(dataset):
            feature = feature.unsqueeze(0)
            output = model(feature)

            hit_probs[i] = F.softmax(output, dim=-1).detach().cpu().numpy()[0]

    return hit_probs


def optimize_hit_score(hit_probs, fps):
    fnum = len(hit_probs)
    if fnum == 1:
        return (
            np.amax(hit_probs),
            [np.argmax(hit_probs)],
            1 if np.argmax(hit_probs) != 0 else 0,
        )

    # Ignore the last second
    if fnum > fps:
        fnum -= fps

    # dp[numframes][3]. i.e. df[f][0] stores the score when label 0 is assigned to
    # current frame f, tgt with hitlabels, numhits
    dp = {}
    dp[0] = {}
    dp[0][0] = (hit_probs[0, 0], [0], 0)
    dp[0][1] = (hit_probs[0, 1], [1], 1)
    dp[0][2] = (hit_probs[0, 2], [2], 1)

    for f in range(1, fnum):
        dp[f] = {}

        # Get frame indices of positive hit labels
        hit_labels = [
            (i, label) for i, label in enumerate(dp[f - 1][0][1]) if label != 0
        ]
        last_hit_label = (-0.5 * fps, 0) if len(hit_labels) == 0 else hit_labels[-1]

        # Assigning not hit to current frame
        dp[f][0] = max(
            [
                (
                    dp[f - 1][0][0] + hit_probs[f, 0],
                    dp[f - 1][0][1] + [0],
                    dp[f - 1][0][2],
                ),
                (
                    dp[f - 1][1][0] + hit_probs[f, 0],
                    dp[f - 1][1][1] + [0],
                    dp[f - 1][1][2],
                ),
                (
                    dp[f - 1][2][0] + hit_probs[f, 0],
                    dp[f - 1][2][1] + [0],
                    dp[f - 1][2][2],
                ),
            ],
            key=lambda x: x[0],
        )

        # 2 constraints: (1) consec hits must be 0.5s apart, (2) hits must alternate between players

        # Assigning near_hit to current frame
        if (f - last_hit_label[0] > 0.5 * fps) and (last_hit_label[1] != 1):
            dp[f][1] = (
                dp[f - 1][0][0] + hit_probs[f, 1],
                dp[f - 1][0][1] + [1],
                dp[f - 1][0][2] + 1,
            )
        else:
            dp[f][1] = (0, [], 0)

        # Assigning far_hit to current frame
        if (f - last_hit_label[0] > 0.5 * fps) and (last_hit_label[1] != 2):
            dp[f][2] = (
                dp[f - 1][0][0] + hit_probs[f, 2],
                dp[f - 1][0][1] + [2],
                dp[f - 1][0][2] + 1,
            )
        else:
            dp[f][2] = (0, [], 0)

    return (
        dp[fnum - 1][0][0],
        np.array(dp[fnum - 1][0][1]).astype("int"),
        dp[fnum - 1][0][2],
    )


def main(args):
    data_id = f"{args.data_id:05d}"
    offset = args.concat_n - 1

    cap = cv2.VideoCapture(os.path.join(args.dataset_path, data_id, f"{data_id}.mp4"))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Prepare dataset and model
    features = preprocess_data(
        args.dataset_path,
        data_id,
        args.concat_n,
        stepsize=1,
        data_type="test",
    )
    dataset = HitNetDataset(features)
    model = HitNet(input_size=features.shape[-1]).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # Run inference
    hit_probs = inference(model, dataset).cpu().numpy()

    # Print out raw predictions
    for hit_prob in hit_probs:
        if np.max(hit_prob) < args.conf_threshold:
            hit_prob[:] = [1, 0, 0]
    print(np.where(np.argmax(hit_probs, axis=1) > 0)[0] + offset)

    # Print out optimized predictions
    hit_score, hit_preds, num_hits = optimize_hit_score(hit_probs, fps)
    print(hit_score, num_hits)
    print(np.where(np.array(hit_preds) > 0)[0] + offset)

    # Save video with visualized hits
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            os.path.join(args.dataset_path, data_id, f"{data_id}_hit.mp4"),
            fourcc,
            fps,
            (width, height),
        )

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx > offset and frame_idx - offset < len(hit_preds):
                curr_pred = hit_preds[frame_idx - offset]
            else:
                curr_pred = 0

            if curr_pred == 1:
                frame[:, :, 0] = 255
            elif curr_pred == 2:
                frame[:, :, 2] = 255

            frame_idx += 1
            out.write(frame)

        cap.release()
        out.release()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--data_id", type=int, required=True)
    parser.add_argument("--concat_n", type=int, default=14)
    parser.add_argument("--conf_threshold", type=float, default=0.9)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--save_video", action="store_true")

    main(parser.parse_args())
