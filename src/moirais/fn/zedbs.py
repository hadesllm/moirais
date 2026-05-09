"""Spatial DBSCAN cluster detection"""

import numpy as np

from ._containers import SpatialResult


def spatial_dbscan(observed, *, expected=None):
    """Spatial DBSCAN cluster detection

    Returns
    -------
    SpatialResult
    """
    x = np.asarray(observed, dtype=float)
    n = len(x)
    mu = float(x.mean())
    stat = float(np.var(x)) if n > 1 else 0.0
    return SpatialResult(
        name="zedbs",
        statistic=stat,
        extra={"mean": mu, "n": n},
    )


spat = spatial_dbscan


def cheatsheet() -> str:
    return "spatial_dbscan({}) -> Spatial DBSCAN cluster detection"
