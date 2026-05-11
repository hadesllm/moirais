# morie.fn — function file (hadesllm/morie)
"""Defection test spatial coalition."""

import numpy as np

from ._containers import SpatialResult


def cfdef(positions, weights, *, threshold=0.5):
    """Defection test spatial coalition.

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
        name="Defection test spatial coalition",
        statistic=float(stat),
        extra={},
    )


cfdef = cfdef  # alias


def cheatsheet() -> str:
    return "cfdef({}) -> Defection test spatial coalition."
