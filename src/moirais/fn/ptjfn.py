# moirais.fn — function file (hadesllm/moirais)
"""J-function (ratio F/G)"""

import numpy as np

from ._containers import SpatialResult


def j_function(data, *, method="default"):
    """J-function (ratio F/G)

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
        name="J-function (ratio F/G)",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


j_fu = j_function


def cheatsheet() -> str:
    return "j_function({}) -> J-function (ratio F/G)"
