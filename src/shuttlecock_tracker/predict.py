# Derived from: https://github.com/wolfyeva/TrackNetV2/blob/main/predict.py

import cv2
import torch
from utils import *


def predict(video_file, model_file):
    num_frame = 3
    batch_size = 10

    checkpoint = torch.load(model_file)
    param_dict = checkpoint["param_dict"]
    model_name = param_dict["model_name"]
    num_frame = param_dict["num_frame"]
    input_type = param_dict["input_type"]

    # Load model
    model = get_model(model_name, num_frame, input_type).cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Cap configuration
    cap = cv2.VideoCapture(video_file)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    success = True
    frame_count = 0
    num_final_frame = 0
    ratio = h / HEIGHT

    # Shuttlecock's coordinates
    coords = []

    while success:
        print(f"Number of sampled frames: {frame_count}")
        # Sample frames to form input sequence
        frame_queue = []
        for _ in range(num_frame * batch_size):
            success, frame = cap.read()
            if not success:
                break
            else:
                frame_count += 1
                frame_queue.append(frame)

        if not frame_queue:
            break

        # If mini batch incomplete
        if len(frame_queue) % num_frame != 0:
            frame_queue = []
            # Record the length of remain frames
            num_final_frame = len(frame_queue)
            # Adjust the sample timestampe of cap
            frame_count = frame_count - num_frame * batch_size
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            # Re-sample mini batch
            for _ in range(num_frame * batch_size):
                success, frame = cap.read()
                if not success:
                    break
                else:
                    frame_count += 1
                    frame_queue.append(frame)
            assert len(frame_queue) % num_frame == 0

        x = get_frame_unit(frame_queue, num_frame)

        # Inference
        with torch.no_grad():
            y_pred = model(x.cuda())
        y_pred = y_pred.detach().cpu().numpy()
        h_pred = y_pred > 0.5
        h_pred = h_pred * 255.0
        h_pred = h_pred.astype("uint8")
        h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)

        for i in range(h_pred.shape[0]):
            if num_final_frame > 0 and i < (num_frame * batch_size - num_final_frame):
                # Special case of last incomplete mini batch
                # Igore the frame which is already written to the output video
                continue
            else:
                img = frame_queue[i].copy()
                cx_pred, cy_pred = get_object_center(h_pred[i])
                cx_pred, cy_pred = int(ratio * cx_pred), int(ratio * cy_pred)
                vis = True if cx_pred > 0 and cy_pred > 0 else False

                coords.append((cx_pred, cy_pred) if vis else (np.nan, np.nan))

    print("Done!")

    return np.array(coords)
