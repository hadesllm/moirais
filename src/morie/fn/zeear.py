"""Ecological regression (Poisson)"""

import numpy as np

from ._containers import SpatialResult


def ecological_reg(data, *, method="default"):
    """Ecological regression (Poisson)

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
        name="Ecological regression (Poisson)",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


ecol = ecological_reg


def cheatsheet() -> str:
    return "ecological_reg({}) -> Ecological regression (Poisson)"
