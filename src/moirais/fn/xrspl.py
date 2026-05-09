"""Spatial logit model"""

import numpy as np

from ._containers import SpatialResult


def spatial_logit(data, *, method="default"):
    """Spatial logit model

    Returns
    -------
    SpatialResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    mu = float(x.mean())
    stat = float(np.var(x)) if n > 1 else 0.0
    return SpatialResult(
        name="xrspl",
        statistic=stat,
        extra={"mean": mu, "n": n},
    )


spat = spatial_logit


def cheatsheet() -> str:
    return "spatial_logit({}) -> Spatial logit model"
