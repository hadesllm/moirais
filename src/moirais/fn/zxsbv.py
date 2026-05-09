"""Spatial block cross-validation"""

import numpy as np

from ._containers import DescriptiveResult


def spatial_cv_block(data, *, method="default"):
    """Spatial block cross-validation

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    val = float(np.var(x)) if n > 1 else 0.0
    return DescriptiveResult(
        name="zxsbv",
        value=val,
        extra={"n": n},
    )


spat = spatial_cv_block


def cheatsheet() -> str:
    return "spatial_cv_block({}) -> Spatial block cross-validation"
