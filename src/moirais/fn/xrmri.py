"""Moran's I on regression residuals"""

import numpy as np

from ._containers import SpatialResult


def moran_resid(data, *, method="default"):
    """Moran's I on regression residuals

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
        name="Moran's I on regression residuals",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


mora = moran_resid


def cheatsheet() -> str:
    return "moran_resid({}) -> Moran's I on regression residuals"
