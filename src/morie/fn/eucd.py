# morie.fn — function file (hadesllm/morie)
"""Euclidean distance."""

import numpy as np

from ._containers import ESRes

_QUOTE = "Difficulties strengthen the mind, as labor does the body. — Seneca"


def euclidean_dist(x, y, **kwargs) -> ESRes:
    r"""
    Compute Euclidean (L2) distance between two vectors.

    .. math::

        d(x, y) = \\sqrt{\\sum_{i=1}^{n} (x_i - y_i)^2}

    :param x: array-like, first vector.
    :param y: array-like, second vector.
    :return: ESRes with Euclidean distance.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have same length.")
    d = float(np.sqrt(np.sum((x - y) ** 2)))
    return ESRes(measure="euclidean_distance", estimate=d, extra={"squared": d**2, "dim": len(x)})


eucd = euclidean_dist


def cheatsheet() -> str:
    return "euclidean_dist({}) -> Euclidean distance."
