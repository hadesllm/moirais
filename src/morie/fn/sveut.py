"""Exponential decay utility."""

import numpy as np

from ._containers import SpatialResult


def sveut(ideal, pos, *, decay=1.0):
    """Exponential decay utility.

    Returns
    -------
    SpatialResult
    """

    ideal = np.asarray(ideal, dtype=float)
    pos = np.asarray(pos, dtype=float)
    d = float(np.linalg.norm(ideal - pos))
    stat = float(np.exp(-decay * d))
    return SpatialResult(
        name="Exponential Decay Utility",
        statistic=float(stat),
        extra={"distance": d, "decay": decay},
    )


sveut = sveut  # alias


def cheatsheet() -> str:
    return "sveut({}) -> Exponential decay utility."
