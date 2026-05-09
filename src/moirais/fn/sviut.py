"""Ideal-point utility deviation."""

import numpy as np

from ._containers import SpatialResult


def sviut(ideal, pos):
    """Ideal-point utility deviation.

    Returns
    -------
    SpatialResult
    """

    ideal = np.asarray(ideal, dtype=float)
    pos = np.asarray(pos, dtype=float)
    dev = float(np.linalg.norm(ideal - pos))
    stat = -dev
    return SpatialResult(
        name="Ideal-Point Deviation Utility",
        statistic=float(stat),
        extra={"deviation": dev},
    )


sviut = sviut  # alias


def cheatsheet() -> str:
    return "sviut({}) -> Ideal-point utility deviation."
