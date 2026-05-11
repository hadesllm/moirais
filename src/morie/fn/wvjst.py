"""Johnston power index."""

import numpy as np

from ._containers import SpatialResult


def wvjst(weights, *, quota=None):
    """Johnston power index.

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
        name="Johnston power index",
        statistic=float(stat),
        extra={},
    )


wvjst = wvjst  # alias


def cheatsheet() -> str:
    return "wvjst({}) -> Johnston power index."
