# moirais.fn — function file (hadesllm/moirais)
"""First-order point pattern stats"""

import numpy as np

from ._containers import SpatialResult


def first_order_pp(data, *, method="default"):
    """First-order point pattern stats

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
        name="First-order point pattern stats",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


firs = first_order_pp


def cheatsheet() -> str:
    return "first_order_pp({}) -> First-order point pattern stats"
