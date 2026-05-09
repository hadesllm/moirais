"""Spatial SIR diffusion"""

import numpy as np

from ._containers import SpatialResult


def spatial_sir(data, *, method="default"):
    """Spatial SIR diffusion

    Returns
    -------
    SpatialResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    mu = float(x.mean())
    stat = float(np.var(x)) if n > 1 else 0.0
    return SpatialResult(
        name="zesir",
        statistic=stat,
        extra={"mean": mu, "n": n},
    )


spat = spatial_sir


def cheatsheet() -> str:
    return "spatial_sir({}) -> Spatial SIR diffusion"
