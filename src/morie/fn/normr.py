# morie.fn — function file (hadesllm/morie)
"""Normalized redundancy."""

import numpy as np

from ._containers import ESRes


def normalized_redundancy(x, y, bins: int = 50, **kwargs) -> ESRes:
    """
    Compute normalised redundancy R(X,Y) = I(X;Y) / min(H(X), H(Y)).

    Measures the fraction of entropy shared between X and Y,
    normalised to [0, 1].

    :param x: array-like, first variable.
    :param y: array-like, second variable.
    :param bins: Number of histogram bins.
    :return: ESRes with normalised redundancy in [0,1].

    References
    ----------
    Studholme C et al. (1999). An overlap invariant entropy measure
    of 3D medical image alignment. Pattern Recognition, 32(1), 71-86.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have same length.")
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    def _h1d(arr):
        counts, _ = np.histogram(arr, bins=bins)
        p = counts / counts.sum()
        p = p[p > 0]
        return -float(np.sum(p * np.log2(p)))

    h_x = _h1d(x)
    h_y = _h1d(y)

    hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = hist_xy / hist_xy.sum()
    p_xy = p_xy[p_xy > 0]
    h_xy = -float(np.sum(p_xy * np.log2(p_xy)))

    mi = h_x + h_y - h_xy
    denom = min(h_x, h_y)
    nr = mi / denom if denom > 0 else 0.0

    return ESRes(
        measure="normalized_redundancy",
        estimate=float(nr),
        n=n,
        extra={"MI": mi, "H_X": h_x, "H_Y": h_y},
    )


normr = normalized_redundancy


def cheatsheet() -> str:
    return "normalized_redundancy(x, y) -> I(X;Y)/min(H(X),H(Y))."
