# moirais.fn — function file (hadesllm/moirais)
"""Homogeneous Poisson point process"""

import numpy as np

from ._containers import SpatialResult


def poisson_process(data, *, method="default"):
    """Homogeneous Poisson point process

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
        name="Homogeneous Poisson point process",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


pois = poisson_process


def cheatsheet() -> str:
    return "poisson_process({}) -> Homogeneous Poisson point process"
