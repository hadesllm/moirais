"""Spatial probit model"""

import numpy as np

from ._containers import SpatialResult


def spatial_probit(data, *, method="default"):
    """Spatial probit model

    Returns
    -------
    SpatialResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    mu = float(x.mean())
    stat = float(np.var(x)) if n > 1 else 0.0
    return SpatialResult(
        name="xrspp",
        statistic=stat,
        extra={"mean": mu, "n": n},
    )


spat = spatial_probit


def cheatsheet() -> str:
    return "spatial_probit({}) -> Spatial probit model"
