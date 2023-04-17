"""
This script uses Tracknet to track shuttlecock in a dataset of videos.
"""

import argparse
import os

import cv2
import numpy as np
from predict import predict
from smooth_trajectory import smooth_trajectory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--load_weights", type=str, required=True)
    parser.add_argument("--save_video", action="store_true")
    return parser.parse_args()


def plot_shuttle_cock(coords, video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (1280, 720))

    for coord in coords:
        ret, frame = cap.read()
        if not ret:
            break

        if not np.isnan(coord).any():
            cv2.circle(frame, (int(coord[0]), int(coord[1])), 5, (0, 0, 255), -1)

        out.write(frame)

    cap.release()
    out.release()


def main(args):
    dataset_path = args.dataset_path
    load_weights = args.load_weights
    save_video = args.save_video

    for split in ["val", "test"]:
        split_path = os.path.join(dataset_path, split)

        for data in sorted(os.listdir(split_path)):
            data_path = os.path.join(split_path, data)

            video_path = os.path.join(data_path, f"{data}.mp4")

            coords = predict(video_path, load_weights)
            coords = smooth_trajectory(coords)

            output_file = f"{data}_trajectory.csv"
            output_path = os.path.join(data_path, output_file)

            with open(output_path, "w") as f:
                f.write("Frame,Visibility,X,Y\n")
                for frame, coord in enumerate(coords):
                    if np.isnan(coord).any():
                        f.write(f"{frame},0,0,0\n")
                    else:
                        f.write(f"{frame},1,{int(coord[0])},{int(coord[1])}\n")

            if save_video:
                plot_shuttle_cock(
                    coords, video_path, output_path.replace(".csv", ".mp4")
                )


if __name__ == "__main__":
    args = parse_args()
    main(args)
