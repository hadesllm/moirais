# morie.fn — function file (hadesllm/morie)
"""Empty space F-function"""

import numpy as np

from ._containers import SpatialResult


def f_function(data, *, method="default"):
    """Empty space F-function

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
        name="Empty space F-function",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


f_fu = f_function


def cheatsheet() -> str:
    return "f_function({}) -> Empty space F-function"
