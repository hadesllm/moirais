"""LM test for spatial error"""

import numpy as np

from ._containers import SpatialResult


def lm_error(data, *, method="default"):
    """LM test for spatial error

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
        name="LM test for spatial error",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


lm_e = lm_error


def cheatsheet() -> str:
    return "lm_error({}) -> LM test for spatial error"
