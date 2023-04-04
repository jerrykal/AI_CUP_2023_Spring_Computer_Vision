import os

import cv2
import pandas as pd

data_dir = os.path.join(os.path.dirname(__file__), "dataset/rally/train")

hit_frames_dir = os.path.join(os.path.dirname(__file__), "dataset/hit_frames/")
if not os.path.exists(hit_frames_dir):
    os.mkdir(hit_frames_dir)

for idx in sorted(os.listdir(data_dir)):
    csvfile_path = os.path.join(data_dir, idx, idx + "_S2.csv")
    csvfile = pd.read_csv(csvfile_path)

    # Extract shot sequence and hit frames
    shot_seq = csvfile["ShotSeq"].values
    hit_frames = csvfile["HitFrame"].values

    video_path = os.path.join(data_dir, idx, f"{idx}.mp4")
    print(f"Extracting hit frames from {idx}.mp4 ... ", end="", flush=True)

    # Open video and extract frames from hit frames
    cap = cv2.VideoCapture(video_path)
    for i in range(len(hit_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, hit_frames[i])
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(
                os.path.join(hit_frames_dir, f"{idx}_{shot_seq[i]:02d}.jpg"), frame
            )
    print("Done!")
