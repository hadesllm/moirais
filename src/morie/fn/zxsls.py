"""Spatial LASSO"""

import numpy as np

from ._containers import DescriptiveResult


def spatial_lasso(data, *, method="default"):
    """Spatial LASSO

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    val = float(np.var(x)) if n > 1 else 0.0
    return DescriptiveResult(
        name="zxsls",
        value=val,
        extra={"n": n},
    )


spat = spatial_lasso


def cheatsheet() -> str:
    return "spatial_lasso({}) -> Spatial LASSO"
