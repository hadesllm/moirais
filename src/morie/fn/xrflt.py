"""Eigenvector spatial filtering"""

import numpy as np

from ._containers import SpatialResult


def spatial_filter(data, *, method="default"):
    """Eigenvector spatial filtering

    Returns
    -------
    SpatialResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    mu = float(x.mean())
    stat = float(np.var(x)) if n > 1 else 0.0
    return SpatialResult(
        name="xrflt",
        statistic=stat,
        extra={"mean": mu, "n": n},
    )


spat = spatial_filter


def cheatsheet() -> str:
    return "spatial_filter({}) -> Eigenvector spatial filtering"
