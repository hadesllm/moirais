"""Probit spatial vote probability."""

import numpy as np

from ._containers import SpatialResult


def svpbp(voter, cA, cB=None, cdf=None, *, beta=1.0):
    """Probit spatial vote probability.

    Returns
    -------
    SpatialResult
    """

    voter = np.asarray(voter, dtype=float)
    cA = np.asarray(cA, dtype=float)
    if cB is None:
        cB = np.zeros_like(cA)
    else:
        cB = np.asarray(cB, dtype=float)
    from scipy.stats import norm as _norm

    dA = float(np.sum((voter - cA) ** 2))
    dB = float(np.sum((voter - cB) ** 2))
    stat = float(_norm.cdf(beta * (dB - dA)))
    return SpatialResult(
        name="Probit Spatial Vote Probability",
        statistic=float(stat),
        extra={"prob_A": stat, "dist_A": dA, "dist_B": dB},
    )


svpbp = svpbp  # alias


def cheatsheet() -> str:
    return "svpbp({}) -> Probit spatial vote probability."
