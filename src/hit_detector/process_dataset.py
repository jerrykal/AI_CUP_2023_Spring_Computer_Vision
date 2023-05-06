import math
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
import torch
from dataset import HitNetDataset, preprocess_data
from detect import inference, optimize_hit_score

from model import HitNet


def main(args):
    offset = math.ceil(args.concat_n * 0.75) - 1

    for data_id in sorted(os.listdir(args.dataset_path)):
        print(f"Processing {data_id}... ", end="", flush=True)

        data_path = os.path.join(args.dataset_path, data_id)

        video_path = os.path.join(data_path, f"{data_id}.mp4")
        cap = cv2.VideoCapture(video_path)
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

        # Inference and optimize result
        hit_probs = inference(model, dataset, device=args.device)
        for hit_prob in hit_probs:
            if np.max(hit_prob) < args.conf_threshold:
                hit_prob[:] = [1, 0, 0]
        _, hit_preds, num_hits = optimize_hit_score(hit_probs, fps)

        # Save the result
        csv_path = os.path.join(data_path, f"{data_id}_hit.csv")
        df_hits = pd.DataFrame(
            {
                "ShotSeq": np.arange(1, num_hits + 1),
                "HitFrame": np.where(np.array(hit_preds) > 0)[0] + offset,
                "Hitter": [
                    "B" if pred == 1 else "A" for pred in hit_preds[hit_preds > 0]
                ],
            }
        )
        df_hits.to_csv(csv_path, index=False)

        # Save video with visualized hits
        if args.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                os.path.join(data_path, f"{data_id}_hit.mp4"),
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

        print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--concat_n", type=int, default=14)
    parser.add_argument("--conf_threshold", type=float, default=0.9)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--save_video", action="store_true")

    main(parser.parse_args())
