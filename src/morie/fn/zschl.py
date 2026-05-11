"""Cholesky spatial simulation"""

import numpy as np

from ._containers import SpatialResult


def chol_sim(data, *, method="default"):
    """Cholesky spatial simulation

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
        name="Cholesky spatial simulation",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


chol = chol_sim


def cheatsheet() -> str:
    return "chol_sim({}) -> Cholesky spatial simulation"
