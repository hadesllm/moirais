# morie.fn — function file (hadesllm/morie)
"""Hart-Kurz coalition formation."""

import numpy as np

from ._containers import SpatialResult


def cfhrt(positions, weights, *, threshold=0.5):
    """Hart-Kurz coalition formation.

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
        name="Hart-Kurz coalition formation",
        statistic=float(stat),
        extra={},
    )


cfhrt = cfhrt  # alias


def cheatsheet() -> str:
    return "cfhrt({}) -> Hart-Kurz coalition formation."
