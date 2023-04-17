import math

import numpy as np


def filter_missing_coords(coords):
    return coords[~np.isnan(coords).any(axis=1)]


def shortest_distance_to_curve(x, y, curve_coef):
    a = curve_coef[2]
    b = curve_coef[1]
    c = curve_coef[0]

    # Reference: https://mathworld.wolfram.com/Point-QuadraticDistance.html
    x_candidates = np.roots(
        [2 * (a**2), 3 * a * b, b**2 + 2 * a * c - 2 * a * y + 1, c * b - y * b - x]
    )
    curve_func = np.poly1d(curve_coef[::-1])
    return min(
        [
            math.dist([x_candidate, curve_func(x_candidate)], [x, y])
            for x_candidate in x_candidates
        ]
    )


def smooth_trajectory(coords):
    # Mark shifted coordinates
    shifted_coords = []
    for i in range(1, len(coords) - 1):
        if coords[i - 1] is None or coords[i] is None or coords[i + 1] is None:
            continue

        if (
            math.dist(coords[i - 1], coords[i]) > 100
            and math.dist(coords[i], coords[i + 1]) > 100
        ):
            shifted_coords.append(i)

    # Remove shifted coordinates
    coords[shifted_coords] = (np.nan, np.nan)

    # Calculate shortest distance between a point and the curves fitted in the front and back check window
    front_check_dist = np.full(len(coords), np.nan)
    back_check_dist = np.full(len(coords), np.nan)
    for i in range(len(coords) - 7):
        curve_coords = filter_missing_coords(coords[i : i + 7])

        if len(curve_coords) < 3:
            continue

        curve_x = curve_coords[:, 0]
        curve_y = curve_coords[:, 1]
        curve_coef = np.polynomial.polynomial.polyfit(curve_x, curve_y, 2)

        if i != 0 and not np.isnan(coords[i - 1]).any():
            front_check_dist[i - 1] = shortest_distance_to_curve(
                coords[i - 1, 0], coords[i - 1, 1], curve_coef
            )

        if i != len(coords) - 7 and not np.isnan(coords[i + 7]).any():
            back_check_dist[i + 7] = shortest_distance_to_curve(
                coords[i + 7, 0], coords[i + 7, 1], curve_coef
            )

    # Remove points that are too far away from its back and front check window curves
    for i, (f_dist, b_dist) in enumerate(zip(front_check_dist, back_check_dist)):
        if (
            not np.isnan(f_dist)
            and not np.isnan(b_dist)
            and f_dist > 100
            and b_dist > 100
        ):
            coords[i] = (np.nan, np.nan)

    x = coords[:, 0]
    y = coords[:, 1]

    # Curve fitting
    for i, (f_dist, b_dist) in enumerate(zip(front_check_dist, back_check_dist)):
        if f_dist is not None and b_dist is not None and f_dist < 5 and b_dist < 5:
            # Fill missing points in the front check window
            missing = np.isnan(x[i - 7 : i + 1])
            x[i - 7 : i + 1][missing] = np.interp(
                np.nonzero(missing)[0],
                np.nonzero(~missing)[0],
                x[i - 7 : i + 1][~missing],
                left=np.nan,
                right=np.nan,
            )

            curve_coords = filter_missing_coords(coords[i - 7 : i + 1])
            curve_x = curve_coords[:, 0]
            curve_y = curve_coords[:, 1]
            curve_coef = np.polynomial.polynomial.polyfit(curve_x, curve_y, 2)

            y[i - 7 : i + 1][missing] = np.poly1d(curve_coef[::-1])(
                x[i - 7 : i + 1][missing]
            )

            # Fill missing points in the back check window
            missing = np.isnan(x[i : i + 8])
            x[i : i + 8][missing] = np.interp(
                np.nonzero(missing)[0],
                np.nonzero(~missing)[0],
                x[i : i + 8][~missing],
                left=np.nan,
                right=np.nan,
            )

            curve_coords = filter_missing_coords(coords[i : i + 8])
            curve_x = curve_coords[:, 0]
            curve_y = curve_coords[:, 1]
            curve_coef = np.polynomial.polynomial.polyfit(curve_x, curve_y, 2)

            y[i : i + 8][missing] = np.poly1d(curve_coef[::-1])(x[i : i + 8][missing])

    interp_begin = None

    # Interpolation
    for i in range(3, len(coords) - 3):
        if not np.isnan(x[i]):
            continue

        if interp_begin is None:
            if not np.isnan(x[i - 3 : i]).any():
                interp_begin = i - 3
        else:
            if i - interp_begin > 8:
                interp_begin = None

        if interp_begin is not None and not np.isnan(x[i + 1 : i + 4]).any():
            interp_end = i + 4
            missing = np.isnan(x[interp_begin:interp_end])
            x[interp_begin:interp_end][missing] = np.interp(
                np.nonzero(missing)[0],
                np.nonzero(~missing)[0],
                x[interp_begin:interp_end][~missing],
            )

            curve_coords = filter_missing_coords(coords[interp_begin:interp_end])
            curve_x = curve_coords[:, 0]
            curve_y = curve_coords[:, 1]
            curve_coef = np.polynomial.polynomial.polyfit(curve_x, curve_y, 2)

            y[interp_begin:interp_end][missing] = np.poly1d(curve_coef[::-1])(
                x[interp_begin:interp_end][missing]
            )

            interp_begin = None

    return np.stack([np.around(x), np.around(y)], axis=1)
