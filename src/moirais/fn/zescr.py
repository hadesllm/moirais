"""Spatial cure rate model"""

import numpy as np

from ._containers import SpatialResult


def spatial_cure_rate(data, *, method="default"):
    """Spatial cure rate model

    Returns
    -------
    SpatialResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    mu = float(x.mean())
    stat = float(np.var(x)) if n > 1 else 0.0
    return SpatialResult(
        name="zescr",
        statistic=stat,
        extra={"mean": mu, "n": n},
    )


spat = spatial_cure_rate


def cheatsheet() -> str:
    return "spatial_cure_rate({}) -> Spatial cure rate model"
