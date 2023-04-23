# https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2

import time

import cv2
import keras.backend as K
import numpy as np
from keras.layers import *
from keras.models import *
from tensorflow.keras.utils import array_to_img, img_to_array

BATCH_SIZE = 1
HEIGHT = 288
WIDTH = 512
# HEIGHT=360
# WIDTH=640
sigma = 2.5
mag = 1


def genHeatMap(w, h, cx, cy, r, mag):
    if cx < 0 or cy < 0:
        return np.zeros((h, w))
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    heatmap = ((y - (cy + 1)) ** 2) + ((x - (cx + 1)) ** 2)
    heatmap[heatmap <= r**2] = 1
    heatmap[heatmap > r**2] = 0
    return heatmap * mag


# Loss function
def custom_loss(y_true, y_pred):
    loss = (-1) * (
        K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1))
        + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1))
    )
    return K.mean(loss)


def predict(video_name, load_weights):
    model = load_model(load_weights, custom_objects={"custom_loss": custom_loss})
    # model.summary()
    print("Beginning predicting......")

    start = time.time()

    coords = []

    cap = cv2.VideoCapture(video_name)

    success, image1 = cap.read()
    success, image2 = cap.read()
    success, image3 = cap.read()

    ratio = image1.shape[0] / HEIGHT

    while success:
        unit = []
        # Adjust BGR format (cv2) to RGB format (PIL)
        x1 = image1[..., ::-1]
        x2 = image2[..., ::-1]
        x3 = image3[..., ::-1]
        # Convert np arrays to PIL images
        x1 = array_to_img(x1)
        x2 = array_to_img(x2)
        x3 = array_to_img(x3)
        # Resize the images
        x1 = x1.resize(size=(WIDTH, HEIGHT))
        x2 = x2.resize(size=(WIDTH, HEIGHT))
        x3 = x3.resize(size=(WIDTH, HEIGHT))
        # Convert images to np arrays and adjust to channels first
        x1 = np.moveaxis(img_to_array(x1), -1, 0)
        x2 = np.moveaxis(img_to_array(x2), -1, 0)
        x3 = np.moveaxis(img_to_array(x3), -1, 0)
        # Create data
        unit.append(x1[0])
        unit.append(x1[1])
        unit.append(x1[2])
        unit.append(x2[0])
        unit.append(x2[1])
        unit.append(x2[2])
        unit.append(x3[0])
        unit.append(x3[1])
        unit.append(x3[2])
        unit = np.asarray(unit)
        unit = unit.reshape((1, 9, HEIGHT, WIDTH))
        unit = unit.astype("float32")
        unit /= 255
        y_pred = model.predict(unit, batch_size=BATCH_SIZE)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype("float32")
        h_pred = y_pred[0] * 255
        h_pred = h_pred.astype("uint8")
        for i in range(3):
            if np.amax(h_pred[i]) <= 0:
                coords.append((np.nan, np.nan))
            else:
                # h_pred
                (cnts, _) = cv2.findContours(
                    h_pred[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for i in range(len(rects)):
                    area = rects[i][2] * rects[i][3]
                    if area > max_area:
                        max_area_idx = i
                        max_area = area
                target = rects[max_area_idx]
                (cx_pred, cy_pred) = (
                    int(ratio * (target[0] + target[2] / 2)),
                    int(ratio * (target[1] + target[3] / 2)),
                )

                coords.append((cx_pred, cy_pred))
        success, image1 = cap.read()
        success, image2 = cap.read()
        success, image3 = cap.read()

    end = time.time()
    print("Prediction time:", end - start, "secs")
    print("Done......")

    return np.array(coords)
