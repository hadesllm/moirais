"""Cauchy Spatial Vote Probability."""

import numpy as np

from ._containers import SpatialResult


def svclp2(voter, cA, cB, *, beta=1.0):
    """Cauchy Spatial Vote Probability.

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
    stat = float(0.5 + np.arctan(z) / np.pi)
    _extra = {"prob_A": stat, "dist_A": dA, "dist_B": dB}

    return SpatialResult(
        name="Cauchy Spatial Vote Probability",
        statistic=float(stat),
        extra=_extra,
    )


svclp2 = svclp2  # alias


def cheatsheet() -> str:
    return "svclp2({}) -> Cauchy Spatial Vote Probability."
