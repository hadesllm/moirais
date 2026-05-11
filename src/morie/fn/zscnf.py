"""Filled contour generation"""

import numpy as np

from ._containers import SpatialResult


def contour_fill(data, *, method="default"):
    """Filled contour generation

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
        name="Filled contour generation",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


cont = contour_fill


def cheatsheet() -> str:
    return "contour_fill({}) -> Filled contour generation"
