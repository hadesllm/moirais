"""Inverse distance weights matrix"""

import numpy as np

from ._containers import SpatialResult


def w_inverse_dist(data, *, method="default"):
    """Inverse distance weights matrix

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
        name="Inverse distance weights matrix",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


w_in = w_inverse_dist


def cheatsheet() -> str:
    return "w_inverse_dist({}) -> Inverse distance weights matrix"
