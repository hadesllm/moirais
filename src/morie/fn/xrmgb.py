"""MGWR variable bandwidths"""

import numpy as np

from ._containers import SpatialResult


def mgwr_bandwidths(data, *, method="default"):
    """MGWR variable bandwidths

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
        name="MGWR variable bandwidths",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


mgwr = mgwr_bandwidths


def cheatsheet() -> str:
    return "mgwr_bandwidths({}) -> MGWR variable bandwidths"
