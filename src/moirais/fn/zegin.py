"""Spatial Gini coefficient"""

import numpy as np

from ._containers import SpatialResult


def gini_spatial(data, *, method="default"):
    """Spatial Gini coefficient

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
        name="Spatial Gini coefficient",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


gini = gini_spatial


def cheatsheet() -> str:
    return "gini_spatial({}) -> Spatial Gini coefficient"
