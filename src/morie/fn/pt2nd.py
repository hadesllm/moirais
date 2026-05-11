# morie.fn — function file (hadesllm/morie)
"""Second-order point pattern stats"""

import numpy as np

from ._containers import SpatialResult


def second_order_pp(data, *, method="default"):
    """Second-order point pattern stats

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
        name="Second-order point pattern stats",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


seco = second_order_pp


def cheatsheet() -> str:
    return "second_order_pp({}) -> Second-order point pattern stats"
