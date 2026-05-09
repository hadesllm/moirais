"""Spatial gradient estimation"""

import numpy as np

from ._containers import SpatialResult


def gradient_spatial(data, *, method="default"):
    """Spatial gradient estimation

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
        name="Spatial gradient estimation",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


grad = gradient_spatial


def cheatsheet() -> str:
    return "gradient_spatial({}) -> Spatial gradient estimation"
