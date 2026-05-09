"""Threshold utility step function."""

import numpy as np

from ._containers import SpatialResult


def svtut(ideal, pos, *, threshold=1.0):
    """Threshold utility step function.

    Returns
    -------
    SpatialResult
    """

    ideal = np.asarray(ideal, dtype=float)
    pos = np.asarray(pos, dtype=float)
    d = float(np.linalg.norm(ideal - pos))
    stat = 1.0 if d <= threshold else 0.0
    return SpatialResult(
        name="Threshold Utility",
        statistic=float(stat),
        extra={"distance": d, "threshold": threshold},
    )


svtut = svtut  # alias


def cheatsheet() -> str:
    return "svtut({}) -> Threshold utility step function."
