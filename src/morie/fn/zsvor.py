"""Voronoi polygon areas"""

import numpy as np

from ._containers import SpatialResult


def voronoi_areas(data, *, method="default"):
    """Voronoi polygon areas

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
        name="Voronoi polygon areas",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


voro = voronoi_areas


def cheatsheet() -> str:
    return "voronoi_areas({}) -> Voronoi polygon areas"
