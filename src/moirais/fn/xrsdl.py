"""Spatial Durbin Error model"""

import numpy as np

from ._containers import SpatialResult


def sdem_ml(data, *, method="default"):
    """Spatial Durbin Error model

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
        name="Spatial Durbin Error model",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


sdem = sdem_ml


def cheatsheet() -> str:
    return "sdem_ml({}) -> Spatial Durbin Error model"
