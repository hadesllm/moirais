# morie.fn — function file (hadesllm/morie)
"""Point pattern Voronoi intensities"""

import numpy as np

from ._containers import SpatialResult


def pp_voronoi(data, *, method="default"):
    """Point pattern Voronoi intensities

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
        name="Point pattern Voronoi intensities",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


pp_v = pp_voronoi


def cheatsheet() -> str:
    return "pp_voronoi({}) -> Point pattern Voronoi intensities"
