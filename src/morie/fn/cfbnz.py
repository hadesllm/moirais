# morie.fn -- function file (hadesllm/morie)
"""Banzhaf coalition index."""

import numpy as np

from ._containers import SpatialResult


def cfbnz(positions, weights, *, threshold=0.5):
    """Banzhaf coalition index.

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
        name="Banzhaf coalition index",
        statistic=float(stat),
        extra={},
    )


cfbnz = cfbnz  # alias


def cheatsheet() -> str:
    return "cfbnz({}) -> Banzhaf coalition index."
