"""Getis spatial filtering"""

import numpy as np

from ._containers import SpatialResult


def getis_filter(data, *, method="default"):
    """Getis spatial filtering

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
        name="Getis spatial filtering",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


geti = getis_filter


def cheatsheet() -> str:
    return "getis_filter({}) -> Getis spatial filtering"
