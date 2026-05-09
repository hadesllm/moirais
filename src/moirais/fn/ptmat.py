# moirais.fn — function file (hadesllm/moirais)
"""Matern cluster process"""

import numpy as np

from ._containers import SpatialResult


def matern_process(data, *, method="default"):
    """Matern cluster process

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
        name="Matern cluster process",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


mate = matern_process


def cheatsheet() -> str:
    return "matern_process({}) -> Matern cluster process"
