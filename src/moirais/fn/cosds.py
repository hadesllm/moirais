# moirais.fn — function file (hadesllm/moirais)
"""Cosine distance."""

import numpy as np

from ._containers import ESRes

_QUOTE = "Rebellions are built on hope. -- Jyn Erso"


def cosine_distance(x, y, **kwargs) -> ESRes:
    """
    Compute cosine distance d = 1 - cos(x, y).

    .. math::

        d(x, y) = 1 - \\frac{x \\cdot y}{\\|x\\| \\, \\|y\\|}

    :param x: array-like, first vector.
    :param y: array-like, second vector.
    :return: ESRes with cosine distance in [0, 2].
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have same length.")
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0 or ny == 0:
        raise ValueError("Zero-norm vector has undefined cosine.")
    sim = float(np.dot(x, y) / (nx * ny))
    d = 1.0 - sim
    return ESRes(measure="cosine_distance", estimate=d, extra={"cosine_similarity": sim, "dim": len(x)})


cosds = cosine_distance


def cheatsheet() -> str:
    return "cosine_distance({}) -> Cosine distance."
