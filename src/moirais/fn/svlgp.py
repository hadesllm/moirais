"""Logit spatial vote probability."""

import numpy as np

from ._containers import SpatialResult


def svlgp(voter, cA, cB=None, *, beta=1.0):
    """Logit spatial vote probability.

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
    from scipy.special import expit

    dA = float(np.sum((voter - cA) ** 2))
    dB = float(np.sum((voter - cB) ** 2))
    stat = float(expit(beta * (dB - dA)))
    return SpatialResult(
        name="Logit Spatial Vote Probability",
        statistic=float(stat),
        extra={"prob_A": stat, "dist_A": dA, "dist_B": dB},
    )


svlgp = svlgp  # alias


def cheatsheet() -> str:
    return "svlgp({}) -> Logit spatial vote probability."
