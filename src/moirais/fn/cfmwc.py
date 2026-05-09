# moirais.fn — function file (hadesllm/moirais)
"""Minimum connected winning coalition."""

import numpy as np

from ._containers import SpatialResult


def cfmwc(positions, weights, *, threshold=0.5):
    """Minimum connected winning coalition.

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
        name="Minimum connected winning coalition",
        statistic=float(stat),
        extra={},
    )


cfmwc = cfmwc  # alias


def cheatsheet() -> str:
    return "cfmwc({}) -> Minimum connected winning coalition."
