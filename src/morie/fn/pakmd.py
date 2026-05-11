# morie.fn — function file (hadesllm/morie)
"""Kemeny-Young aggregation."""

import numpy as np

from ._containers import SpatialResult


def pakmd(rankings):
    """Kemeny-Young aggregation.

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
        name="Kemeny-Young aggregation",
        statistic=float(stat),
        extra={},
    )


pakmd = pakmd  # alias


def cheatsheet() -> str:
    return "pakmd({}) -> Kemeny-Young aggregation."
