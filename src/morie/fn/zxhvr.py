"""Haversine distance"""

import numpy as np

from ._containers import DescriptiveResult


def haversine_dist(data, *, method="default"):
    """Haversine distance

    Returns
    -------
    DescriptiveResult
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if n > 0 else 0.0
    return DescriptiveResult(
        name="zxhvr",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


have = haversine_dist


def cheatsheet() -> str:
    return "haversine_dist({}) -> Haversine distance"
