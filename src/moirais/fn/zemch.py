"""Maternal-child health mapping"""

import numpy as np

from ._containers import SpatialResult


def maternal_child_map(data, *, method="default"):
    """Maternal-child health mapping

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
        name="Maternal-child health mapping",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


mate = maternal_child_map


def cheatsheet() -> str:
    return "maternal_child_map({}) -> Maternal-child health mapping"
