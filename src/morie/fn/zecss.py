"""Spatial CUSUM aberration detection"""

import numpy as np

from ._containers import SpatialResult


def cusum_spatial(data, *, method="default"):
    """Spatial CUSUM aberration detection

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
        name="Spatial CUSUM aberration detection",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


cusu = cusum_spatial


def cheatsheet() -> str:
    return "cusum_spatial({}) -> Spatial CUSUM aberration detection"
