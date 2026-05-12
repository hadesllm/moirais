# morie.fn -- function file (hadesllm/morie)
"""Mutual information."""

import numpy as np

from ._containers import ESRes

_QUOTE = "Let the past die. Kill it if you have to. -- Kylo Ren"


def mutual_information(x, y, bins: int = 50, **kwargs) -> ESRes:
    """
    Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).

    .. math::

        I(X;Y) = H(X) + H(Y) - H(X,Y)

    :param x: array-like, first variable.
    :param y: array-like, second variable.
    :param bins: Number of bins per dimension.
    :return: ESRes with MI in bits.

    References
    ----------
    Cover TM, Thomas JA (2006). Elements of Information Theory,
    2nd ed. Wiley, New York.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have same length.")

    def _entropy(arr):
        c, _ = np.histogram(arr, bins=bins)
        p = c / c.sum()
        p = p[p > 0]
        return -float(np.sum(p * np.log2(p)))

    h_x = _entropy(x)
    h_y = _entropy(y)
    hist_joint, _, _ = np.histogram2d(x, y, bins=bins)
    pj = hist_joint / hist_joint.sum()
    pj_pos = pj[pj > 0]
    h_xy = -float(np.sum(pj_pos * np.log2(pj_pos)))
    mi = h_x + h_y - h_xy
    return ESRes(
        measure="mutual_information",
        estimate=max(mi, 0.0),
        n=len(x),
        extra={"h_x": h_x, "h_y": h_y, "h_xy": h_xy, "bits": max(mi, 0.0)},
    )


mutif = mutual_information


def cheatsheet() -> str:
    return "mutual_information({}) -> Mutual information."
