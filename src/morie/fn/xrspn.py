"""Spatial panel fixed effects"""

import numpy as np

from ._containers import SpatialResult


def spatial_panel_fe(data, *, method="default"):
    """Spatial panel fixed effects

    Returns
    -------
    SpatialResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    mu = float(x.mean())
    stat = float(np.var(x)) if n > 1 else 0.0
    return SpatialResult(
        name="xrspn",
        statistic=stat,
        extra={"mean": mu, "n": n},
    )


spat = spatial_panel_fe


def cheatsheet() -> str:
    return "spatial_panel_fe({}) -> Spatial panel fixed effects"
