"""Spatial GLMM simulation"""

import numpy as np

from ._containers import SpatialResult


def spatial_glmm_sim(data, *, method="default"):
    """Spatial GLMM simulation

    Returns
    -------
    SpatialResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    mu = float(x.mean())
    stat = float(np.var(x)) if n > 1 else 0.0
    return SpatialResult(
        name="zsglm",
        statistic=stat,
        extra={"mean": mu, "n": n},
    )


spat = spatial_glmm_sim


def cheatsheet() -> str:
    return "spatial_glmm_sim({}) -> Spatial GLMM simulation"
