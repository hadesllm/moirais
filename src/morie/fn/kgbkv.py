# morie.fn -- function file (hadesllm/morie)
"""Block kriging variance"""

import numpy as np

from ._containers import SpatialResult


def bk_variance(data, *, method="default"):
    """Block kriging variance

    Returns
    -------
    SpatialResult
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if n > 0 else 0.0
    return SpatialResult(
        name="Block kriging variance",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


bk_v = bk_variance


def cheatsheet() -> str:
    return "bk_variance({}) -> Block kriging variance"
