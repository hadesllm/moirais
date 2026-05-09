# moirais.fn — function file (hadesllm/moirais)
"""Correlation distance."""

import numpy as np

from ._containers import ESRes

_QUOTE = "Luck is what happens when preparation meets opportunity. — Seneca"


def correlation_dist(x, y, **kwargs) -> ESRes:
    """
    Compute correlation distance d = 1 - r(x, y).

    .. math::

        d(x, y) = 1 - \\frac{\\text{Cov}(x,y)}{\\sigma_x \\sigma_y}

    :param x: array-like, first vector.
    :param y: array-like, second vector.
    :return: ESRes with correlation distance in [0, 2].
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have same length.")
    if len(x) < 2:
        raise ValueError("Need at least 2 observations.")
    r = float(np.corrcoef(x, y)[0, 1])
    d = 1.0 - r
    return ESRes(measure="correlation_distance", estimate=d, extra={"pearson_r": r, "dim": len(x)})


corrd = correlation_dist


def cheatsheet() -> str:
    return "correlation_dist({}) -> Correlation distance."
