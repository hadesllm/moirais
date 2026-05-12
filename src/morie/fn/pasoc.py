# morie.fn -- function file (hadesllm/morie)
"""Social choice spatial."""

import numpy as np

from ._containers import SpatialResult


def pasoc(rankings):
    """Social choice spatial.

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
        name="Social choice spatial",
        statistic=float(stat),
        extra={},
    )


pasoc = pasoc  # alias


def cheatsheet() -> str:
    return "pasoc({}) -> Social choice spatial."
