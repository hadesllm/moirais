"""Hierarchical spatial (nested)"""

import numpy as np

from ._containers import DescriptiveResult


def hier_spatial_fe(data, *, method="default"):
    """Hierarchical spatial (nested)

    Returns
    -------
    DescriptiveResult
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if n > 0 else 0.0
    return DescriptiveResult(
        name="zxhrs",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


hier = hier_spatial_fe


def cheatsheet() -> str:
    return "hier_spatial_fe({}) -> Hierarchical spatial (nested)"
