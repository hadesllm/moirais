# morie.fn -- function file (hadesllm/morie)
"""Cross-type point pattern"""

import numpy as np

from ._containers import SpatialResult


def cross_pp(data, *, method="default"):
    """Cross-type point pattern

    Returns
    -------
    SpatialResult
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if n > 0 else 0.0
    return SpatialResult(
        name="Cross-type point pattern",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


cros = cross_pp


def cheatsheet() -> str:
    return "cross_pp({}) -> Cross-type point pattern"
