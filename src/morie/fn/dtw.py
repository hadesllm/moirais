# morie.fn — function file (hadesllm/morie)
"""Dynamic time warping distance."""

import numpy as np

from ._containers import ESRes


def dtw_distance(x: np.ndarray, y: np.ndarray) -> ESRes:
    """
    Compute the Dynamic Time Warping (DTW) distance between two sequences.

    Uses dynamic programming to find the optimal alignment that
    minimises the cumulative Euclidean distance.

    :param x: (n,) first time series.
    :param y: (m,) second time series.
    :return: ESRes with DTW distance as estimate.

    References
    ----------
    Sakoe H, Chiba S (1978). Dynamic programming algorithm optimization
    for spoken word recognition. IEEE Transactions on Acoustics, Speech,
    and Signal Processing, 26(1), 43-49.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n, m = len(x), len(y)
    dtw_mat = np.full((n + 1, m + 1), np.inf)
    dtw_mat[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            dtw_mat[i, j] = cost + min(dtw_mat[i - 1, j], dtw_mat[i, j - 1], dtw_mat[i - 1, j - 1])
    dist = float(np.sqrt(dtw_mat[n, m]))
    normalised = dist / (n + m)
    return ESRes(measure="dtw_distance", estimate=dist, extra={"normalised": normalised, "n": n, "m": m})


dtw = dtw_distance


def cheatsheet() -> str:
    return "dtw_distance({}) -> Dynamic time warping distance."
