"""Power-weighted utility function."""

import numpy as np

from ._containers import SpatialResult


def svpwt(ideal, pos, *, power=2.0):
    """Power-weighted utility function.

    Returns
    -------
    SpatialResult
    """

    ideal = np.asarray(ideal, dtype=float)
    pos = np.asarray(pos, dtype=float)
    d = float(np.linalg.norm(ideal - pos))
    stat = -(d**power)
    return SpatialResult(
        name="Power-Weighted Utility",
        statistic=float(stat),
        extra={"distance": d, "power": power},
    )


svpwt = svpwt  # alias


def cheatsheet() -> str:
    return "svpwt({}) -> Power-weighted utility function."
