# morie.fn -- function file (hadesllm/morie)
"""Mutual information I(X;Y)."""

import numpy as np

from ._containers import ESRes


def mutual_information(x, y, bins: int = 50, **kwargs) -> ESRes:
    """
    Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).

    .. math::

        I(X;Y) = H(X) + H(Y) - H(X,Y)

    :param x: array-like, first variable.
    :param y: array-like, second variable.
    :param bins: Number of histogram bins per dimension.
    :return: ESRes with mutual information in bits.

    References
    ----------
    Cover TM, Thomas JA (2006). Elements of Information Theory,
    2nd ed. Wiley, New York.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have same length.")
    if len(x) < 2:
        raise ValueError("Need at least 2 observations.")

    def _entropy_1d(arr):
        counts, _ = np.histogram(arr, bins=bins)
        p = counts / counts.sum()
        p = p[p > 0]
        return -float(np.sum(p * np.log2(p)))

    h_x = _entropy_1d(x)
    h_y = _entropy_1d(y)

    hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = hist_xy / hist_xy.sum()
    p_xy = p_xy[p_xy > 0]
    h_xy = -float(np.sum(p_xy * np.log2(p_xy)))

    mi = h_x + h_y - h_xy
    return ESRes(
        measure="mutual_information",
        estimate=mi,
        n=len(x),
        extra={"H_X": h_x, "H_Y": h_y, "H_XY": h_xy, "n_bins": bins},
    )


minfo = mutual_information


def cheatsheet() -> str:
    return "mutual_information(x, y) -> Mutual information I(X;Y)."
