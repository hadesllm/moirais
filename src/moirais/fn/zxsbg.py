"""Spatial bagging"""

import numpy as np

from ._containers import DescriptiveResult


def spatial_bagging(data, *, method="default"):
    """Spatial bagging

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    val = float(np.var(x)) if n > 1 else 0.0
    return DescriptiveResult(
        name="zxsbg",
        value=val,
        extra={"n": n},
    )


spat = spatial_bagging


def cheatsheet() -> str:
    return "spatial_bagging({}) -> Spatial bagging"
