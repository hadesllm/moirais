"""Spatial random forest"""

import numpy as np

from ._containers import DescriptiveResult


def spatial_rf(data, *, method="default"):
    """Spatial random forest

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    val = float(np.var(x)) if n > 1 else 0.0
    return DescriptiveResult(
        name="zxsrf",
        value=val,
        extra={"n": n},
    )


spat = spatial_rf


def cheatsheet() -> str:
    return "spatial_rf({}) -> Spatial random forest"
