"""Hotspot detection map"""

import numpy as np

from ._containers import SpatialResult


def hotspot_map(data, *, method="default"):
    """Hotspot detection map

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
        name="Hotspot detection map",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


hots = hotspot_map


def cheatsheet() -> str:
    return "hotspot_map({}) -> Hotspot detection map"
