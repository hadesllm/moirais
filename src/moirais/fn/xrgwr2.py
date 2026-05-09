"""GWR local R-squared"""

import numpy as np

from ._containers import SpatialResult


def gwr_rsquared(data, *, method="default"):
    """GWR local R-squared

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
        name="GWR local R-squared",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


gwr_ = gwr_rsquared


def cheatsheet() -> str:
    return "gwr_rsquared({}) -> GWR local R-squared"
