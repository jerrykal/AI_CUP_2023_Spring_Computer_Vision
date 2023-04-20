import itertools

import cv2
import numpy as np
from utils import (check_white, filter_set, find_edge, find_intersection,
                   get_background, get_hough_lines, get_white_line_region,
                   interpolate, lines_clustering, merge_lines)


def detect_court_corners(video_path):
    # Construct background model
    cap = cv2.VideoCapture(video_path)
    image = get_background(cap)
    cap.release()

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Find court region
    green = np.uint8([[[0, 255, 0]]])
    hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    lower_green = hsv_green - [20, 255, 200]
    upper_green = hsv_green + [30, 0, 0]

    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.erode(mask.astype(np.float32), None, iterations=3)
    mask = cv2.dilate(mask.astype(np.float32), None, iterations=10)
    mask = cv2.erode(mask, None, iterations=15)
    mask = np.stack((mask, mask, mask), 2)

    # Use Hough transform to find straight lines
    lines = get_hough_lines(gray)

    # Filter out lines
    court_lines = []

    for line in lines.reshape(-1, 4):
        full_line = interpolate(line)
        check = mask[:, :, 0][full_line[:, 1], full_line[:, 0]]
        if check.sum() >= len(check) * 0.5:
            court_lines.append(line)

    # Extend straight lines
    extended_court_lines = []
    for x1, y1, x2, y2 in court_lines:
        (extended_x1, extended_y1), (extended_x2, extended_y2) = find_edge(
            np.array([x1, y1]), np.array([x2, y2]), height, width
        )
        extended_court_lines.append(
            np.array([extended_x1, extended_y1, extended_x2, extended_y2])
        )
    extended_court_lines.sort(key=lambda x: x[1])

    idx_non_0 = next(
        (
            index
            for index, value in enumerate(np.array(extended_court_lines)[:, 1].tolist())
            if value != 0
        ),
        None,
    )
    idx_is_height = next(
        (
            index
            for index, value in enumerate(np.array(extended_court_lines)[:, 1].tolist())
            if value == height - 1
        ),
        None,
    )
    extended_court_lines[:idx_non_0] = sorted(
        extended_court_lines[:idx_non_0], key=lambda x: -x[0]
    )
    if idx_is_height != None:
        extended_court_lines[idx_is_height:] = sorted(
            extended_court_lines[idx_is_height:], key=lambda x: x[0]
        )
    extended_court_lines = merge_lines(extended_court_lines)

    # Filter out lines based on if the lines satify the line structure contrain
    # See section 3.1 of https://www.researchgate.net/publication/220979520_Robust_camera_calibration_for_sport_videos_using_court_models
    white_line_region = get_white_line_region(gray)
    line_set = [
        i
        for i in lines_clustering(extended_court_lines, threshold=8)
        if len(i) > 1 and check_white(hsv, extended_court_lines, i, white_line_region)
    ]

    fake_sets = filter_set(extended_court_lines, line_set, height, width)

    # Classify lines into horizontal and left/right vertical lines
    lines = [sublist[0] for sublist in fake_sets]

    h_lines = []
    v_lines = []

    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx > 2 * dy:
            h_lines.append(i)
        else:
            v_lines.append(i)

    left_v_lines = []
    right_v_lines = []

    left_th = width / 3
    right_th = width * 2 / 3

    for i in v_lines:
        x1, y1, x2, y2 = lines[i]
        if x1 < left_th or x2 < left_th:
            left_v_lines.append(i)
        elif x1 > right_th or x2 > right_th:
            right_v_lines.append(i)

    # Find court corners
    cand = 0
    max_win = 0
    for l1, l2 in list(itertools.combinations(h_lines, 2)):
        for l3 in left_v_lines:
            for l4 in right_v_lines:
                # Make sure that l1 is the top line and l2 is the bottom line
                if lines[l1][1] > lines[l2][1]:
                    l1, l2 = l2, l1

                # Upper left, upper right, lower right, lower left corners
                ul = np.array(find_intersection(lines[l1], lines[l3]))
                ur = np.array(find_intersection(lines[l1], lines[l4]))
                lr = np.array(find_intersection(lines[l2], lines[l4]))
                ll = np.array(find_intersection(lines[l2], lines[l3]))
                pts = np.array([[ur, lr, ul, ll]]).astype(np.int32)

                cand += 1
                test = np.zeros((height, width), dtype=np.uint8)
                test = cv2.fillPoly(test, pts, (1, 1, 1)).astype(bool)

                if (
                    np.logical_and(mask[..., 0], test).sum() > max_win
                    and np.logical_and(mask[..., 0], test).sum() * 100 / test.sum() > 80
                ):
                    print("new winner:")
                    max_win = np.logical_and(mask[..., 0], test).sum()
                    best = ((l1, l2), (l3, l4))

                print(
                    "{} {} count {}/{}, {:.2f}, {}".format(
                        (l1, l2),
                        (l3, l4),
                        np.logical_and(mask[..., 0], test).sum(),
                        test.sum(),
                        np.logical_and(mask[..., 0], test).sum() * 100 / test.sum(),
                        ""
                        if np.logical_and(mask[..., 0], test).sum() * 100 / test.sum()
                        < 80
                        else "good",
                    )
                )

    ((l1, l2), (l3, l4)) = best
    fake_sets[l1].sort(key=lambda x: x[1])
    fake_sets[l2].sort(key=lambda x: x[1])
    fake_sets[l3].sort(key=lambda x: x[0])
    fake_sets[l4].sort(key=lambda x: x[0])
    ul = np.array(find_intersection(fake_sets[l1][0], fake_sets[l3][0]), dtype=np.int32)
    ur = np.array(find_intersection(fake_sets[l1][0], fake_sets[l4][1]), dtype=np.int32)
    lr = np.array(find_intersection(fake_sets[l2][1], fake_sets[l4][1]), dtype=np.int32)
    ll = np.array(find_intersection(fake_sets[l2][1], fake_sets[l3][0]), dtype=np.int32)

    # cv2.circle(image, (ul[0], ul[1]), 5, (255, 0, 0), -1)
    # cv2.circle(image, (ur[0], ur[1]), 5, (0, 255, 0), -1)
    # cv2.circle(image, (lr[0], lr[1]), 5, (0, 0, 255), -1)
    # cv2.circle(image, (ll[0], ll[1]), 5, (0, 255, 255), -1)
    # cv2.imshow("Court Corners", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return ul, ur, lr, ll
