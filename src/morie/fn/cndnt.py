# morie.fn — function file (hadesllm/morie)
"""Conditional entropy H(X|Y)."""

import numpy as np

from ._containers import ESRes


def conditional_entropy(x, y, bins: int = 50, **kwargs) -> ESRes:
    """
    Compute conditional entropy H(X|Y) = H(X,Y) - H(Y).

    .. math::

        H(X|Y) = H(X,Y) - H(Y)

    :param x: array-like, target variable.
    :param y: array-like, conditioning variable.
    :param bins: Number of histogram bins per dimension.
    :return: ESRes with conditional entropy in bits.

    References
    ----------
    Cover TM, Thomas JA (2006). Elements of Information Theory,
    2nd ed. Wiley, New York.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have same length.")
    if len(x) < 1:
        raise ValueError("Need at least 1 observation.")
    hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = hist_xy / hist_xy.sum()
    p_xy_pos = p_xy[p_xy > 0]
    h_xy = -float(np.sum(p_xy_pos * np.log2(p_xy_pos)))

    counts_y, _ = np.histogram(y, bins=bins)
    p_y = counts_y / counts_y.sum()
    p_y_pos = p_y[p_y > 0]
    h_y = -float(np.sum(p_y_pos * np.log2(p_y_pos)))

    h_cond = h_xy - h_y
    return ESRes(
        measure="conditional_entropy",
        estimate=h_cond,
        n=len(x),
        extra={"H_XY": h_xy, "H_Y": h_y, "n_bins": bins},
    )


cndnt = conditional_entropy


def cheatsheet() -> str:
    return "conditional_entropy(x, y) -> Conditional entropy H(X|Y)."
