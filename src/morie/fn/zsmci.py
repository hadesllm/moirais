"""Monte Carlo spatial integration"""

import numpy as np

from ._containers import SpatialResult


def mc_spatial_int(data, *, method="default"):
    """Monte Carlo spatial integration

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
        name="Monte Carlo spatial integration",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


mc_s = mc_spatial_int


def cheatsheet() -> str:
    return "mc_spatial_int({}) -> Monte Carlo spatial integration"
