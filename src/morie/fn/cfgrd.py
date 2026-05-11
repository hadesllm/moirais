# morie.fn — function file (hadesllm/morie)
"""Grand coalition spatial."""

import numpy as np

from ._containers import SpatialResult


def cfgrd(positions, weights, *, threshold=0.5):
    """Grand coalition spatial.

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
        name="Grand coalition spatial",
        statistic=float(stat),
        extra={},
    )


cfgrd = cfgrd  # alias


def cheatsheet() -> str:
    return "cfgrd({}) -> Grand coalition spatial."
