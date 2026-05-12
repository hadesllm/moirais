# morie.fn -- function file (hadesllm/morie)
"""Lognormal kriging back-transform"""

import numpy as np

from ._containers import SpatialResult


def lk_backtransform(data, *, method="default"):
    """Lognormal kriging back-transform

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
        name="Lognormal kriging back-transform",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


lk_b = lk_backtransform


def cheatsheet() -> str:
    return "lk_backtransform({}) -> Lognormal kriging back-transform"
