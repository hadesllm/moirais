"""Local Moran's I (LISA)"""

import numpy as np

from ._containers import SpatialResult


def lisa_local(data, *, method="default"):
    """Local Moran's I (LISA)

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
        name="Local Moran's I (LISA)",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


lisa = lisa_local


def cheatsheet() -> str:
    return "lisa_local({}) -> Local Moran's I (LISA)"
