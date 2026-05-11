"""Fuzzy utility with membership degrees."""

import numpy as np

from ._containers import SpatialResult


def svfut(ideal, pos, *, fuzz=1.0):
    """Fuzzy utility with membership degrees.

    Returns
    -------
    SpatialResult
    """

    ideal = np.asarray(ideal, dtype=float)
    pos = np.asarray(pos, dtype=float)
    d = float(np.linalg.norm(ideal - pos))
    membership = float(np.exp(-(d**2) / (2 * fuzz**2)))
    stat = membership
    return SpatialResult(
        name="Fuzzy Membership Utility",
        statistic=float(stat),
        extra={"distance": d, "membership": membership},
    )


svfut = svfut  # alias


def cheatsheet() -> str:
    return "svfut({}) -> Fuzzy utility with membership degrees."
