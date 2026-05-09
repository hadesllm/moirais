# moirais.fn — function file (hadesllm/moirais)
"""Universal kriging variance"""

import numpy as np

from ._containers import SpatialResult


def uk_variance(data, *, method="default"):
    """Universal kriging variance

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
        name="Universal kriging variance",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


uk_v = uk_variance


def cheatsheet() -> str:
    return "uk_variance({}) -> Universal kriging variance"
