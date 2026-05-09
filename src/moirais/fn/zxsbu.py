"""Buffered spatial CV"""

import numpy as np

from ._containers import DescriptiveResult


def spatial_cv_buffer(data, *, method="default"):
    """Buffered spatial CV

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    val = float(np.var(x)) if n > 1 else 0.0
    return DescriptiveResult(
        name="zxsbu",
        value=val,
        extra={"n": n},
    )


spat = spatial_cv_buffer


def cheatsheet() -> str:
    return "spatial_cv_buffer({}) -> Buffered spatial CV"
