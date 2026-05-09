"""Fay-Herriot small area estimator"""

import numpy as np

from ._containers import SpatialResult


def fay_herriot(data, *, method="default"):
    """Fay-Herriot small area estimator

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
        name="Fay-Herriot small area estimator",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


fay_ = fay_herriot


def cheatsheet() -> str:
    return "fay_herriot({}) -> Fay-Herriot small area estimator"
