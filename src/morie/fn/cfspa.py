# morie.fn — function file (hadesllm/morie)
"""Spatial coalition formation."""

import numpy as np

from ._containers import SpatialResult


def cfspa(positions, weights, *, threshold=0.5):
    """Spatial coalition formation.

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
        name="Spatial coalition formation",
        statistic=float(stat),
        extra={},
    )


cfspa = cfspa  # alias


def cheatsheet() -> str:
    return "cfspa({}) -> Spatial coalition formation."
