"""Gravity-based accessibility"""

import numpy as np

from ._containers import SpatialResult


def gravity_access(data, *, method="default"):
    """Gravity-based accessibility

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
        name="Gravity-based accessibility",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


grav = gravity_access


def cheatsheet() -> str:
    return "gravity_access({}) -> Gravity-based accessibility"
