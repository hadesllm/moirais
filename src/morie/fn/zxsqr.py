"""Spatial quantile regression"""

import numpy as np

from ._containers import DescriptiveResult


def spatial_quantile(data, *, method="default"):
    """Spatial quantile regression

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    val = float(np.var(x)) if n > 1 else 0.0
    return DescriptiveResult(
        name="zxsqr",
        value=val,
        extra={"n": n},
    )


spat = spatial_quantile


def cheatsheet() -> str:
    return "spatial_quantile({}) -> Spatial quantile regression"
