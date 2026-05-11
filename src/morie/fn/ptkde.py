# morie.fn — function file (hadesllm/morie)
"""Spatial kernel density estimation"""

import numpy as np

from ._containers import SpatialResult


def spatial_kde(data, *, method="default"):
    """Spatial kernel density estimation

    Parameters
    ----------
    data : array_like
        Input data points.
    method : str
        KDE method (default 'default').

    Returns
    -------
    SpatialResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    bw = 1.06 * np.std(x) * n ** (-0.2) if np.std(x) > 0 else 1.0
    mean_density = float(1.0 / (n * bw * np.sqrt(2 * np.pi)))
    return SpatialResult(
        name="ptkde",
        statistic=mean_density,
        extra={"n": n, "bandwidth": float(bw), "method": method},
    )


spat = spatial_kde


def cheatsheet() -> str:
    return "spatial_kde({}) -> Spatial kernel density estimation"
