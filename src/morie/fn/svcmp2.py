"""Complementary Log-Log Vote Probability."""

import numpy as np

from ._containers import SpatialResult


def svcmp2(voter, cA, cB, *, beta=1.0):
    """Complementary Log-Log Vote Probability.

    Returns
    -------
    SpatialResult
    """
    voter = np.asarray(voter, dtype=float)
    cA = np.asarray(cA, dtype=float)
    cB = np.asarray(cB, dtype=float)
    dA = float(np.sum((voter - cA) ** 2))
    dB = float(np.sum((voter - cB) ** 2))
    z = beta * (dB - dA)
    stat = float(1.0 - np.exp(-np.exp(z)))
    _extra = {"prob_A": stat, "dist_A": dA, "dist_B": dB}

    return SpatialResult(
        name="Complementary Log-Log Vote Probability",
        statistic=float(stat),
        extra=_extra,
    )


svcmp2 = svcmp2  # alias


def cheatsheet() -> str:
    return "svcmp2({}) -> Complementary Log-Log Vote Probability."
