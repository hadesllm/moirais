"""Spatial gradient boosting"""

import numpy as np

from ._containers import DescriptiveResult


def spatial_gbm(data, *, method="default"):
    """Spatial gradient boosting

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    val = float(np.var(x)) if n > 1 else 0.0
    return DescriptiveResult(
        name="zxsgb",
        value=val,
        extra={"n": n},
    )


spat = spatial_gbm


def cheatsheet() -> str:
    return "spatial_gbm({}) -> Spatial gradient boosting"
