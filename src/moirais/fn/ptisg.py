# moirais.fn — function file (hadesllm/moirais)
"""Isotropic edge correction"""

import numpy as np

from ._containers import SpatialResult


def isotropic_guard(data, *, method="default"):
    """Isotropic edge correction

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
        name="Isotropic edge correction",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


isot = isotropic_guard


def cheatsheet() -> str:
    return "isotropic_guard({}) -> Isotropic edge correction"
