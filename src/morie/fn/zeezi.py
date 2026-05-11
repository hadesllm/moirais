"""Ecological zero-inflated"""

import numpy as np

from ._containers import SpatialResult


def ecological_zip(data, *, method="default"):
    """Ecological zero-inflated

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
        name="Ecological zero-inflated",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


ecol = ecological_zip


def cheatsheet() -> str:
    return "ecological_zip({}) -> Ecological zero-inflated"
