"""Proximity with fatigue function."""

import numpy as np

from ._containers import SpatialResult


def svpft(voter, candidates, *, fatigue=0.5):
    """Proximity with fatigue function.

    Returns
    -------
    SpatialResult
    """

    voter = np.asarray(voter, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    dists = np.linalg.norm(candidates - voter, axis=1)
    fatigued = dists ** (1 + fatigue)
    choice = int(np.argmin(fatigued))
    stat = float(fatigued[choice])
    return SpatialResult(
        name="Proximity with Fatigue",
        statistic=float(stat),
        extra={"chosen": choice, "fatigue": fatigue},
    )


svpft = svpft  # alias


def cheatsheet() -> str:
    return "svpft({}) -> Proximity with fatigue function."
