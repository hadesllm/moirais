"""Poisson-Gamma disease mapping"""

import numpy as np

from ._containers import SpatialResult


def disease_map_gamma(data, *, method="default"):
    """Poisson-Gamma disease mapping

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
        name="Poisson-Gamma disease mapping",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


dise = disease_map_gamma


def cheatsheet() -> str:
    return "disease_map_gamma({}) -> Poisson-Gamma disease mapping"
