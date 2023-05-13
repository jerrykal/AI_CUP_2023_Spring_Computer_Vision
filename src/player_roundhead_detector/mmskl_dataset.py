import csv
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


class Config:
    def __init__(self, file_path: str):
        with open(file_path, "r") as stream:
            data = yaml.safe_load(stream)
        self.left_segment_size = data['left_segment_size']
        self.right_segment_size = data['right_segment_size']
        self.dataset_path = data['dataset_path']
        self.train_path = data['train_path']

        # test the dataset path and train path is exist or not
        # if not, raise error
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError("Dataset path not found")
        if not os.path.exists(self.train_path):
            os.makedirs(self.train_path)

class Dataset:
    def __init__(self, dataset_path: str, train_path: str):
        # self.config = config
        self.dataset_path = dataset_path
        self.videos = {}
        self.label  = {}
        for root, dirs, files in os.walk(self.dataset_path):
            for dir in dirs:
                dir_num = int(dir)
                dir_num_str = str(dir_num).zfill(5)
                dir_path = os.path.join(root, dir_num_str)
                video_file = os.path.join(dir_path, dir_num_str + ".mp4")
                label_file = os.path.join(dir_path, dir_num_str + "_S2.csv")
                if os.path.exists(video_file) and os.path.exists(label_file):
                    self.videos[dir_num_str] = video_file
                    self.label[dir_num_str] = label_file

    def preprocess_videos(self, left_segment_size: int, right_segment_size: int, train_path: str):
        # Create the CSV file and write the header
        with open(os.path.join(train_path, "video_data.csv"), "w", newline="") as csvfile:
            fieldnames = ["video_id", "hit_frame", "hitter", "round_head", "back_hand", "ball_height", "segment_file"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for video_id, video_path in self.videos.items():
                # Read the label file
                label_file = os.path.join(os.path.dirname(video_path), f"{video_id}_S2.csv")
                labels = pd.read_csv(label_file)

                # Open the video file
                video = cv2.VideoCapture(video_path)

                # Loop through each hit frame in the CSV file
                for _, row in labels.iterrows():
                    hit_frame = int(row["HitFrame"])
                    hitter = row["Hitter"]
                    round_head = int(row["RoundHead"])
                    back_hand = int(row["Backhand"])
                    ball_height = int(row["BallHeight"])

                    # Calculate the segment range
                    start_frame = max(0, hit_frame - left_segment_size)
                    end_frame = hit_frame + right_segment_size

                    # Process the video to extract the segment
                    segment_frames = []
                    for frame_num in range(start_frame, end_frame + 1):
                        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                        ret, frame = video.read()

                        if ret:
                            segment_frames.append(frame)

                    # Save the segment as a new video file
                    segment_file = os.path.join(train_path, f"{video_id}_{hit_frame}_segment.mp4")
                    height, width, _ = segment_frames[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(segment_file, fourcc, 30.0, (width, height))

                    for frame in segment_frames:
                        out.write(frame)

                    out.release()

                    # Append the video data to the CSV file
                    writer.writerow({
                        "video_id": video_id,
                        "hit_frame": hit_frame,
                        "hitter": hitter,
                        "round_head": round_head,
                        "back_hand": back_hand,
                        "ball_height": ball_height,
                        "segment_file": ntpath.split(segment_file)[1]
                    })
                    csvfile.flush()

                video.release()

config = Config("config.yaml")
# print(config.left_segment_size, config.right_segment_size)
dataset = Dataset(config.dataset_path, config.train_path)
dataset.preprocess_videos(config.left_segment_size, config.right_segment_size, config.train_path)
print(len(dataset.videos))