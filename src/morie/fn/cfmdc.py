# morie.fn — function file (hadesllm/morie)
"""Median coalition spatial."""

import numpy as np

from ._containers import SpatialResult


def cfmdc(positions, weights, *, threshold=0.5):
    """Median coalition spatial.

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
        name="Median coalition spatial",
        statistic=float(stat),
        extra={},
    )


cfmdc = cfmdc  # alias


def cheatsheet() -> str:
    return "cfmdc({}) -> Median coalition spatial."
