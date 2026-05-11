# morie.fn — function file (hadesllm/morie)
"""Approval voting spatial."""

import numpy as np

from ._containers import SpatialResult


def paapp(rankings):
    """Approval voting spatial.

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
        name="Approval voting spatial",
        statistic=float(stat),
        extra={},
    )


paapp = paapp  # alias


def cheatsheet() -> str:
    return "paapp({}) -> Approval voting spatial."
