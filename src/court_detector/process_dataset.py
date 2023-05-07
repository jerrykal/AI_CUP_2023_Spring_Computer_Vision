import argparse
import os

import cv2
from detect import detect_court_corners


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    return parser.parse_args()


def is_amateur(video_path):
    """Check if the video is amateur or professional"""
    cap = cv2.VideoCapture(video_path)

    # Get the middle frame
    len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, len // 2)
    _, img = cap.read()
    cap.release()

    top_left_corner = img[: img.shape[0] // 4, : img.shape[1] // 3, :]

    return top_left_corner.sum() > 30000000


def main(args):
    dataset_path = args.dataset_path

    failed_case = []

    for data in sorted(os.listdir(dataset_path)):
        data_path = os.path.join(dataset_path, data)

        video_path = os.path.join(data_path, f"{data}.mp4")

        if is_amateur(video_path):
            image = None
            ul, ur, lr, ll = [[500, 355], [20, 650], [1215, 615], [790, 350]]
        else:
            try:
                ul, ur, lr, ll, image = detect_court_corners(video_path)
            except KeyboardInterrupt:
                return
            except:
                print(f"Failed to detect court corners for {data}")
                failed_case.append(video_path)
                continue

        output_file = f"{data}_court.csv"
        output_path = os.path.join(data_path, output_file)

        with open(output_path, "w") as f:
            f.write(
                "upper_left_x,upper_left_y,upper_right_x,upper_right_y,lower_right_x,lower_right_y,lower_left_x,lower_left_y\n"
            )
            f.write(
                f"{ul[0]},{ul[1]},{ur[0]},{ur[1]},{lr[0]},{lr[1]},{ll[0]},{ll[1]}\n"
            )

        if image is not None:
            cv2.imwrite(output_path.replace(".csv", ".jpg"), image)

    if len(failed_case) != 0:
        with open("failed_case.txt", "a") as f:
            f.write("\n".join(failed_case) + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
