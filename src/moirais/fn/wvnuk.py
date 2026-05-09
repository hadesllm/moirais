"""Nucleolus spatial voting."""

import numpy as np

from ._containers import SpatialResult


def wvnuk(weights, *, quota=None):
    """Nucleolus spatial voting.

    Returns
    -------
    SpatialResult
    """

    weights = np.asarray(weights, dtype=float)
    if quota is None:
        quota = float(weights.sum()) / 2.0
    else:
        quota = float(quota)
    n = len(weights)
    total = weights.sum()
    swing_count = 0
    for i in range(n):
        without_i = np.delete(weights, i).sum()
        if without_i < quota <= without_i + weights[i]:
            swing_count += 1
    stat = float(swing_count) / n if n > 0 else 0.0
    return SpatialResult(
        name="Nucleolus spatial voting",
        statistic=float(stat),
        extra={},
    )


wvnuk = wvnuk  # alias


def cheatsheet() -> str:
    return "wvnuk({}) -> Nucleolus spatial voting."
