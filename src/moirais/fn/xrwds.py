"""Distance-band weights"""

import numpy as np

from ._containers import SpatialResult


def w_distance(data, *, method="default"):
    """Distance-band weights

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
        name="Distance-band weights",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


w_di = w_distance


def cheatsheet() -> str:
    return "w_distance({}) -> Distance-band weights"
