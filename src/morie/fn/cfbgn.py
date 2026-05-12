# morie.fn -- function file (hadesllm/morie)
"""Bargaining coalition spatial."""

import numpy as np

from ._containers import SpatialResult


def cfbgn(positions, weights, *, threshold=0.5):
    """Bargaining coalition spatial.

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
        name="Bargaining coalition spatial",
        statistic=float(stat),
        extra={},
    )


cfbgn = cfbgn  # alias


def cheatsheet() -> str:
    return "cfbgn({}) -> Bargaining coalition spatial."
