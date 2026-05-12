# morie.fn -- function file (hadesllm/morie)
"""Ideological coalition distance."""

import numpy as np

from ._containers import SpatialResult


def cfico(positions, weights, *, threshold=0.5):
    """Ideological coalition distance.

    Returns
    -------
    SpatialResult
    """

    positions = np.asarray(positions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    threshold = float(threshold)
    total_w = weights.sum()
    n = len(positions)
    best_size = n
    best_pos = float(np.average(positions, weights=weights))
    stat = best_pos
    return SpatialResult(
        name="Ideological coalition distance",
        statistic=float(stat),
        extra={},
    )


cfico = cfico  # alias


def cheatsheet() -> str:
    return "cfico({}) -> Ideological coalition distance."
