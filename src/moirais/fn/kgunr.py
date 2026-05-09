# moirais.fn — function file (hadesllm/moirais)
"""Universal kriging residual"""

import numpy as np

from ._containers import SpatialResult


def uk_residual(data, *, method="default"):
    """Universal kriging residual

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
        name="Universal kriging residual",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


uk_r = uk_residual


def cheatsheet() -> str:
    return "uk_residual({}) -> Universal kriging residual"
