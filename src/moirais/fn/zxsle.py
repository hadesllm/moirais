"""Spatial elastic net"""

import numpy as np

from ._containers import DescriptiveResult


def spatial_elastic(data, *, method="default"):
    """Spatial elastic net

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    val = float(np.var(x)) if n > 1 else 0.0
    return DescriptiveResult(
        name="zxsle",
        value=val,
        extra={"n": n},
    )


spat = spatial_elastic


def cheatsheet() -> str:
    return "spatial_elastic({}) -> Spatial elastic net"
