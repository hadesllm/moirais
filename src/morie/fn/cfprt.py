# morie.fn -- function file (hadesllm/morie)
"""Pareto-optimal coalition."""

import numpy as np

from ._containers import SpatialResult


def cfprt(positions, weights, *, threshold=0.5):
    """Pareto-optimal coalition.

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
        name="Pareto-optimal coalition",
        statistic=float(stat),
        extra={},
    )


cfprt = cfprt  # alias


def cheatsheet() -> str:
    return "cfprt({}) -> Pareto-optimal coalition."
