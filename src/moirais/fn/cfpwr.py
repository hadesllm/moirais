# moirais.fn — function file (hadesllm/moirais)
"""Coalition power distribution."""

import numpy as np

from ._containers import SpatialResult


def cfpwr(ideal, pos, power=2.0):
    """Coalition power distribution.

    Parameters
    ----------
    ideal : array-like
        Ideal point coordinates.
    pos : array-like
        Position coordinates.
    power : float
        Power parameter (default 2.0).

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


cfpwr = cfpwr  # alias


def cheatsheet() -> str:
    return "cfpwr({}) -> Coalition power distribution."
