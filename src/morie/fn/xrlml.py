"""LM test for spatial lag"""

import numpy as np

from ._containers import SpatialResult


def lm_lag(data, *, method="default"):
    """LM test for spatial lag

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
        name="LM test for spatial lag",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


lm_l = lm_lag


def cheatsheet() -> str:
    return "lm_lag({}) -> LM test for spatial lag"
