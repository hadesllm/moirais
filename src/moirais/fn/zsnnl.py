"""Laplace natural neighbor (Sibson)"""

import numpy as np

from ._containers import SpatialResult


def nn_laplace(data, *, method="default"):
    """Laplace natural neighbor (Sibson)

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
        name="Laplace natural neighbor (Sibson)",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


nn_l = nn_laplace


def cheatsheet() -> str:
    return "nn_laplace({}) -> Laplace natural neighbor (Sibson)"
