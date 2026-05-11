"""Indirect standardization"""

import numpy as np

from ._containers import SpatialResult


def indirect_std(data, *, method="default"):
    """Indirect standardization

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
        name="Indirect standardization",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


indi = indirect_std


def cheatsheet() -> str:
    return "indirect_std({}) -> Indirect standardization"
