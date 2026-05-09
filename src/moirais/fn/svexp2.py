"""Exponential Vote Probability Decay."""

import numpy as np

from ._containers import SpatialResult


def svexp2(voter, cA, cB, *, beta=1.0):
    """Exponential Vote Probability Decay.

    Returns
    -------
    SpatialResult
    """
    voter = np.asarray(voter, dtype=float)
    cA = np.asarray(cA, dtype=float)
    cB = np.asarray(cB, dtype=float)
    dA = float(np.linalg.norm(voter - cA))
    dB = float(np.linalg.norm(voter - cB))
    stat = float(np.exp(-beta * dA) / (np.exp(-beta * dA) + np.exp(-beta * dB) + 1e-9))
    _extra = {"prob_A": stat, "dist_A": dA, "dist_B": dB}

    return SpatialResult(
        name="Exponential Vote Probability Decay",
        statistic=float(stat),
        extra=_extra,
    )


svexp2 = svexp2  # alias


def cheatsheet() -> str:
    return "svexp2({}) -> Exponential Vote Probability Decay."
