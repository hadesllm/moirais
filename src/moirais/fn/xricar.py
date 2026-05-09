"""Intrinsic CAR (ICAR)"""

import numpy as np

from ._containers import SpatialResult


def icar_model(data, *, method="default"):
    """Intrinsic CAR (ICAR)

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
        name="Intrinsic CAR (ICAR)",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


icar = icar_model


def cheatsheet() -> str:
    return "icar_model({}) -> Intrinsic CAR (ICAR)"
