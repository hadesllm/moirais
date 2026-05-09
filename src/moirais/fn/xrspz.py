"""Spatial zero-inflated Poisson"""

import numpy as np

from ._containers import SpatialResult


def spatial_zip(data, *, method="default"):
    """Spatial zero-inflated Poisson

    Returns
    -------
    SpatialResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    mu = float(x.mean())
    stat = float(np.var(x)) if n > 1 else 0.0
    return SpatialResult(
        name="xrspz",
        statistic=stat,
        extra={"mean": mu, "n": n},
    )


spat = spatial_zip


def cheatsheet() -> str:
    return "spatial_zip({}) -> Spatial zero-inflated Poisson"
