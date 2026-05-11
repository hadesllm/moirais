"""Spatial stacking ensemble"""

import numpy as np

from ._containers import DescriptiveResult


def spatial_stacking(data, *, method="default"):
    """Spatial stacking ensemble

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    val = float(np.var(x)) if n > 1 else 0.0
    return DescriptiveResult(
        name="zxsen",
        value=val,
        extra={"n": n},
    )


spat = spatial_stacking


def cheatsheet() -> str:
    return "spatial_stacking({}) -> Spatial stacking ensemble"
