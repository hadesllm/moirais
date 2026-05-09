"""Spatial frailty survival model"""

import numpy as np

from ._containers import SpatialResult


def spatial_frailty(data, *, method="default"):
    """Spatial frailty survival model

    Returns
    -------
    SpatialResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    mu = float(x.mean())
    stat = float(np.var(x)) if n > 1 else 0.0
    return SpatialResult(
        name="zesfr",
        statistic=stat,
        extra={"mean": mu, "n": n},
    )


spat = spatial_frailty


def cheatsheet() -> str:
    return "spatial_frailty({}) -> Spatial frailty survival model"
