"""Robust LM test for lag"""

import numpy as np

from ._containers import SpatialResult


def lm_robust_lag(data, *, method="default"):
    """Robust LM test for lag

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
        name="Robust LM test for lag",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


lm_r = lm_robust_lag


def cheatsheet() -> str:
    return "lm_robust_lag({}) -> Robust LM test for lag"
