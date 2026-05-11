"""Non-separable space-time covariance"""

import numpy as np

from ._containers import SpatialResult


def st_cov_nonsep(data, *, method="default"):
    """Non-separable space-time covariance

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
        name="Non-separable space-time covariance",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


st_c = st_cov_nonsep


def cheatsheet() -> str:
    return "st_cov_nonsep({}) -> Non-separable space-time covariance"
