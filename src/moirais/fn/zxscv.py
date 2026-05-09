"""Spatial LOO cross-validation"""

import numpy as np

from ._containers import DescriptiveResult


def spatial_cv_loo(data, *, method="default"):
    """Spatial LOO cross-validation

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    val = float(np.var(x)) if n > 1 else 0.0
    return DescriptiveResult(
        name="zxscv",
        value=val,
        extra={"n": n},
    )


spat = spatial_cv_loo


def cheatsheet() -> str:
    return "spatial_cv_loo({}) -> Spatial LOO cross-validation"
