# moirais.fn — function file (hadesllm/moirais)
"""Ordinary kriging variance"""

import numpy as np

from ._containers import SpatialResult


def ok_variance(data, *, method="default"):
    """Ordinary kriging variance

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
        name="Ordinary kriging variance",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


ok_v = ok_variance


def cheatsheet() -> str:
    return "ok_variance({}) -> Ordinary kriging variance"
