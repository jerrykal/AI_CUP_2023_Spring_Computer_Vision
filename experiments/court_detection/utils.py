from math import atan, pi

import cv2
import numpy as np


def check_white(hsv, lines, set_i, white_region):
    #     print('Set')
    #     print(set_i)

    line_mask = np.zeros(hsv.shape[:2], "uint8")
    for line_id in set_i:
        x1, y1, x2, y2 = lines[line_id]
        cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 1)
    line_mask = cv2.dilate(line_mask.astype(np.float32), None, iterations=2) > 0

    white_line_region = np.logical_and(line_mask, white_region)
    #     cv2.imshow('white_line_region', white_line_region.astype('uint8')*255)
    #     cv2.waitKey(0)

    cv2.destroyAllWindows()
    #     print((white_line_region.sum() / line_mask.sum()))
    return (white_line_region.sum() / line_mask.sum()) > 0.03


def find_edge(p1, p2, h, w):
    v12 = p2 - p1 + 1e-18
    target = np.array([0 if v12[0] < 0 else w - 1, 0 if v12[1] < 0 else h - 1])
    steps = (target - p2) / v12
    edge_p2 = (steps.min() * v12 + p2).astype(int)

    v21 = -v12
    target = np.array([0 if v21[0] < 0 else w - 1, 0 if v21[1] < 0 else h - 1])
    steps = (target - p1) / v21
    edge_p1 = (steps.min() * v21 + p1).astype(int)
    if edge_p1[1] == 0:  # start with top edge
        return edge_p1.astype(int), edge_p2.astype(int)
    elif edge_p1[0] == 0:  # start with left edge
        if edge_p2[1] == 0:  # end with top edge
            return edge_p2.astype(int), edge_p1.astype(int)
        else:
            return edge_p1.astype(int), edge_p2.astype(int)
    elif edge_p1[1] == h - 1:  # start with botton edge
        if edge_p2[0] == w - 1:  # end with right edge
            return edge_p1.astype(int), edge_p2.astype(int)
        else:
            return edge_p2.astype(int), edge_p1.astype(int)
    elif edge_p1[0] == w - 1:  # start with right edge
        return edge_p2.astype(int), edge_p1.astype(int)


def find_intersection(l1, l2):
    with np.errstate(divide="ignore", invalid="ignore"):
        # s * (x - px) + py = y
        s1 = (l1[3] - l1[1]) / (l1[2] - l1[0])
        s2 = (l2[3] - l2[1]) / (l2[2] - l2[0])
        s1 = 1e20 if np.isinf(s1) else s1
        s2 = 1e20 if np.isinf(s2) else s2
        px1 = l1[0]
        py1 = l1[1]
        px2 = l2[0]
        py2 = l2[1]
        # [s1 * (x - px1) + py1] - [s2 * (x - px2) + py2] = 0
        x = (-py1 + py2 + s1 * px1 - s2 * px2) / (s1 - s2)
        x = 1e6 if np.isnan(x) else x
        y = s1 * (x - px1) + py1 if (x - px1) != 0 else s2 * (x - px2) + py2
        return (x, y)


def merge_lines(lines):
    num_l = len(lines)
    #     remove = []
    for l1 in range(num_l):
        for l2 in range(l1 + 1, num_l):
            if (lines[l1] == lines[l2]).all() or (
                lines[l1] == np.roll(lines[l2], 2)
            ).all():
                #                 print(lines[l1], lines[l2])
                #                 print(lines[l1], np.roll(lines[l2], 2))
                lines[l2] = np.array([0, 0, 0, 0])
    lines = [i for i in lines if not (i == np.array([0, 0, 0, 0])).all()]
    return lines


def dist(samples):
    xx, xy = np.meshgrid(samples[:, 0], samples[:, 0])
    yx, yy = np.meshgrid(samples[:, 1], samples[:, 1])
    ans = np.sqrt((xy - xx) ** 2 + (yx - yy) ** 2)
    xx, xy = np.meshgrid(samples[:, 2], samples[:, 2])
    yx, yy = np.meshgrid(samples[:, 3], samples[:, 3])
    ans = np.maximum(ans, np.sqrt((xy - xx) ** 2 + (yx - yy) ** 2))

    ans = np.triu(ans)
    ans[ans == 0] = ans.max() + 1
    return ans


def lines_clustering(lines: list, threshold, distance=dist):
    def step(data, group):
        if len(group) == 1:
            return
        differ_map = distance(np.array(data))
        min_idx = np.unravel_index(differ_map.argmin(), differ_map.shape)
        if differ_map[min_idx] > threshold:
            return
        data[min_idx[0]] = (data[min_idx[0]] + data[min_idx[1]]) / 2
        group[min_idx[0]] = group[min_idx[0]].union(group[min_idx[1]])
        data.pop(min_idx[1])
        group.pop(min_idx[1])
        del differ_map
        step(data, group)

    lines0 = lines.copy()
    #     print('data:', lines0)
    group0 = [set([i]) for i in range(len(lines0))]
    #     print('init:', group0)

    step(lines0, group0)
    #     print('ans:', group0)
    group1 = []
    for set_i in group0:
        slopes = []
        print(set_i)
        for idx, line_i in enumerate(set_i):
            (x1, y1, x2, y2) = lines[line_i]
            slopes.append(
                [
                    line_i,
                    atan((y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 1e6) + pi / 2,
                ]
            )
        #             slopes.sort(key=lambda i:i[1])

        ele = np.array(slopes)[:, 1]
        mean = np.mean(ele, axis=0)
        sd = np.std(ele, axis=0)
        #         print('MEAN_STD', mean, sd)
        newset = [x for x in slopes if (x[1] >= mean - 2 * sd)]
        newset = [x for x in newset if (x[1] <= mean + 2 * sd)]
        if len(newset) > 1:
            print("slopes:", slopes)
            print("newset:", newset)
            newset = [x[0] for x in newset]
            group1.append(set(newset))
    print(group1)
    return group1


def filter_set(lines, sets, _h, _w):
    fake_lines_sets = []
    lines = np.array(lines)
    for seti in sets:
        lines_s_in_set = lines[list(seti), :2]
        lines_e_in_set = lines[list(seti), 2:]
        s_min = lines_s_in_set.sum(1).argmin()
        s_max = lines_s_in_set.sum(1).argmax()
        e_min = lines_e_in_set.sum(1).argmin()
        e_max = lines_e_in_set.sum(1).argmax()
        p1 = lines_s_in_set[s_min]
        p2 = lines_s_in_set[s_max]
        p3 = lines_e_in_set[e_min]
        p4 = lines_e_in_set[e_max]
        inter = find_intersection(
            np.concatenate((p1, p3), axis=0), np.concatenate((p2, p4), axis=0)
        )
        if inter[0] < 0 or inter[0] >= _w or inter[1] < 0 or inter[1] >= _h:
            if p1[1] == 0:
                if p1.sum() > p2.sum():
                    fake_lines_sets.append(
                        [
                            np.concatenate((p1, p3), axis=0),
                            np.concatenate((p2, p4), axis=0),
                        ]
                    )
                else:
                    fake_lines_sets.append(
                        [
                            np.concatenate((p2, p4), axis=0),
                            np.concatenate((p1, p3), axis=0),
                        ]
                    )
            else:
                if p1.sum() > p2.sum():
                    fake_lines_sets.append(
                        [
                            np.concatenate((p2, p4), axis=0),
                            np.concatenate((p1, p3), axis=0),
                        ]
                    )
                else:
                    fake_lines_sets.append(
                        [
                            np.concatenate((p1, p3), axis=0),
                            np.concatenate((p2, p4), axis=0),
                        ]
                    )
        else:
            #             fake_lines_sets.append([np.concatenate((p1, p4), axis=0), np.concatenate((p2, p3), axis=0)])
            if p1[1] == 0:
                if p1.sum() > p2.sum():
                    fake_lines_sets.append(
                        [
                            np.concatenate((p1, p4), axis=0),
                            np.concatenate((p2, p3), axis=0),
                        ]
                    )
                else:
                    fake_lines_sets.append(
                        [
                            np.concatenate((p2, p3), axis=0),
                            np.concatenate((p1, p4), axis=0),
                        ]
                    )
            else:
                if p1.sum() > p2.sum():
                    fake_lines_sets.append(
                        [
                            np.concatenate((p2, p3), axis=0),
                            np.concatenate((p1, p4), axis=0),
                        ]
                    )
                else:
                    fake_lines_sets.append(
                        [
                            np.concatenate((p1, p4), axis=0),
                            np.concatenate((p2, p3), axis=0),
                        ]
                    )
    return fake_lines_sets


def parse_color(strcolor):
    strcolor = strcolor[4:-1].split(",")
    strcolor.reverse()
    return tuple(map(int, strcolor))
