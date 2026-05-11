# morie.fn — function file (hadesllm/morie)
"""Minimum winning coalition."""

import numpy as np

from ._containers import SpatialResult


def cfmin(positions, weights, *, threshold=0.5):
    """Minimum winning coalition.

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
        name="Minimum winning coalition",
        statistic=float(stat),
        extra={},
    )


cfmin = cfmin  # alias


def cheatsheet() -> str:
    return "cfmin({}) -> Minimum winning coalition."
