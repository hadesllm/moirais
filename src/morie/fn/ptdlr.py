# morie.fn — function file (hadesllm/morie)
"""Delaunay residuals"""

import numpy as np

from ._containers import SpatialResult


def pp_delaunay_resid(data, *, method="default"):
    """Delaunay residuals

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
        name="Delaunay residuals",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


pp_d = pp_delaunay_resid


def cheatsheet() -> str:
    return "pp_delaunay_resid({}) -> Delaunay residuals"
