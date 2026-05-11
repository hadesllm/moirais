"""Carstairs deprivation index"""

import numpy as np

from ._containers import SpatialResult


def carstairs_index(data, *, method="default"):
    """Carstairs deprivation index

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
        name="Carstairs deprivation index",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


cars = carstairs_index


def cheatsheet() -> str:
    return "carstairs_index({}) -> Carstairs deprivation index"
