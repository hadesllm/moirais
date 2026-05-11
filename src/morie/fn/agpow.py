# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Agenda power index."""

import numpy as np

from ._containers import SpatialResult


def agpow(ideal, pos, power=2.0):
    """Agenda power index.

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


agpow = agpow  # alias


def cheatsheet() -> str:
    return "agpow({}) -> Agenda power index."
