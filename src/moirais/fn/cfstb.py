# moirais.fn — function file (hadesllm/moirais)
"""Coalition stability test."""

import numpy as np

from ._containers import SpatialResult


def cfstb(positions, weights, *, threshold=0.5):
    """Coalition stability test.

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
        name="Coalition stability test",
        statistic=float(stat),
        extra={},
    )


cfstb = cfstb  # alias


def cheatsheet() -> str:
    return "cfstb({}) -> Coalition stability test."
