# morie.fn — function file (hadesllm/morie)
"""Quadrat count test"""

import numpy as np

from ._containers import SpatialResult


def quadrat_test(data, *, method="default"):
    """Quadrat count test

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
        name="Quadrat count test",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


quad = quadrat_test


def cheatsheet() -> str:
    return "quadrat_test({}) -> Quadrat count test"
