"""Kernel density exposure"""

import numpy as np

from ._containers import SpatialResult


def kernel_exposure(data, *, method="default"):
    """Kernel density exposure

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
        name="Kernel density exposure",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


kern = kernel_exposure


def cheatsheet() -> str:
    return "kernel_exposure({}) -> Kernel density exposure"
