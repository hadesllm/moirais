# morie.fn -- function file (hadesllm/morie)
"""Shapley coalition value."""

import numpy as np

from ._containers import SpatialResult


def cfshp(positions, weights, *, threshold=0.5):
    """Shapley coalition value.

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
        name="Shapley coalition value",
        statistic=float(stat),
        extra={},
    )


cfshp = cfshp  # alias


def cheatsheet() -> str:
    return "cfshp({}) -> Shapley coalition value."
