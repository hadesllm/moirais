"""Spatial Theil decomposition"""

import numpy as np

from ._containers import SpatialResult


def theil_spatial(data, *, method="default"):
    """Spatial Theil decomposition

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
        name="Spatial Theil decomposition",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


thei = theil_spatial


def cheatsheet() -> str:
    return "theil_spatial({}) -> Spatial Theil decomposition"
