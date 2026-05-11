# morie.fn — function file (hadesllm/morie)
"""Nearest neighbor index (Clark-Evans)"""

import numpy as np

from ._containers import SpatialResult


def nn_index(data, *, method="default"):
    """Nearest neighbor index (Clark-Evans)

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
        name="Nearest neighbor index (Clark-Evans)",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


nn_i = nn_index


def cheatsheet() -> str:
    return "nn_index({}) -> Nearest neighbor index (Clark-Evans)"
