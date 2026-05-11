"""Local Getis-Ord Gi*"""

import numpy as np

from ._containers import SpatialResult


def lisa_getis(data, *, method="default"):
    """Local Getis-Ord Gi*

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
        name="Local Getis-Ord Gi*",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


lisa = lisa_getis


def cheatsheet() -> str:
    return "lisa_getis({}) -> Local Getis-Ord Gi*"
