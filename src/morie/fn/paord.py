# morie.fn — function file (hadesllm/morie)
"""Ordinalist preference aggregation."""

import numpy as np

from ._containers import SpatialResult


def paord(rankings):
    """Ordinalist preference aggregation.

    Returns
    -------
    SpatialResult
    """

    rankings = np.asarray(rankings, dtype=float)
    n_voters, n_options = rankings.shape
    scores = np.zeros(n_options)
    for i in range(n_options):
        for j in range(n_options):
            if i != j:
                scores[i] += float(np.sum(rankings[:, i] < rankings[:, j]))
    winner = int(np.argmax(scores))
    stat = float(scores[winner])
    return SpatialResult(
        name="Ordinalist preference aggregation",
        statistic=float(stat),
        extra={},
    )


paord = paord  # alias


def cheatsheet() -> str:
    return "paord({}) -> Ordinalist preference aggregation."
