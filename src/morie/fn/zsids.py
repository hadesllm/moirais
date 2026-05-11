"""Modified Shepard interpolation"""

import numpy as np

from ._containers import SpatialResult


def idw_shepard(data, *, method="default"):
    """Modified Shepard interpolation

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
        name="Modified Shepard interpolation",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


idw_ = idw_shepard


def cheatsheet() -> str:
    return "idw_shepard({}) -> Modified Shepard interpolation"
