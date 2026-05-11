# morie.fn — function file (hadesllm/morie)
"""Minkowski distance."""

import numpy as np

from ._containers import ESRes

_QUOTE = "The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert"


def minkowski_dist(x, y, p: float = 2.0, **kwargs) -> ESRes:
    """
    Compute Minkowski distance of order p.

    .. math::

        d(x, y) = \\left( \\sum_{i=1}^{n} |x_i - y_i|^p \\right)^{1/p}

    :param x: array-like, first vector.
    :param y: array-like, second vector.
    :param p: Order of the distance (p >= 1). p=1 is Manhattan, p=2 is Euclidean.
    :return: ESRes with Minkowski distance.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have same length.")
    if p < 1:
        raise ValueError("p must be >= 1.")
    d = float(np.sum(np.abs(x - y) ** p) ** (1.0 / p))
    return ESRes(measure="minkowski_distance", estimate=d, extra={"p": p, "dim": len(x)})


minkd = minkowski_dist


def cheatsheet() -> str:
    return "minkowski_dist({}) -> Minkowski distance."
