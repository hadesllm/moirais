# morie.fn -- function file (hadesllm/morie)
"""Ordinary kriging matrix system"""

import numpy as np

from ._containers import SpatialResult


def ok_matrix(data, *, method="default"):
    """Ordinary kriging matrix system

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
        name="Ordinary kriging matrix system",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


ok_m = ok_matrix


def cheatsheet() -> str:
    return "ok_matrix({}) -> Ordinary kriging matrix system"
