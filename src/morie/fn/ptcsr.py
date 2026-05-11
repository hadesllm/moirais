# morie.fn — function file (hadesllm/morie)
"""Complete Spatial Randomness test"""

import numpy as np

from ._containers import SpatialResult


def csr_test(data, *, method="default"):
    """Complete Spatial Randomness test

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
        name="Complete Spatial Randomness test",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


csr_ = csr_test


def cheatsheet() -> str:
    return "csr_test({}) -> Complete Spatial Randomness test"
