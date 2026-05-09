"""Proximity-directional 2D comparison."""

import numpy as np

from ._containers import SpatialResult


def svpd2(voter, cA, cB, *, alpha=0.5):
    """Proximity-directional 2D comparison.

    Returns
    -------
    SpatialResult
    """

    voter = np.asarray(voter, dtype=float)
    cA = np.asarray(cA, dtype=float)
    cB = np.asarray(cB, dtype=float)
    prox_A = -float(np.linalg.norm(voter - cA))
    prox_B = -float(np.linalg.norm(voter - cB))
    dir_A = float(np.dot(voter, cA))
    dir_B = float(np.dot(voter, cB))
    score_A = alpha * prox_A + (1 - alpha) * dir_A
    score_B = alpha * prox_B + (1 - alpha) * dir_B
    stat = score_A - score_B
    return SpatialResult(
        name="Proximity-Directional 2D Comparison",
        statistic=float(stat),
        extra={"score_A": score_A, "score_B": score_B, "prefers_A": stat > 0},
    )


svpd2 = svpd2  # alias


def cheatsheet() -> str:
    return "svpd2({}) -> Proximity-directional 2D comparison."
