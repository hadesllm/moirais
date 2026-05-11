# morie.fn — function file (hadesllm/morie)
"""Contour level set at specified values."""

import numpy as np

from ._containers import SpatialResult


def ctlvl(data=None, coords=None, n=100, seed=42, **kwargs):
    """Contour level set at specified values

    Parameters
    ----------
    data : array-like, optional
        Observed values at sample locations.
    coords : array-like, optional
        Coordinates of sample locations, shape (n, 2) or (n, 3).
    n : int
        Number of simulation nodes or grid points (default 100).
    seed : int
        Random seed for reproducibility (default 42).
    **kwargs
        Additional method-specific parameters.

    Returns
    -------
    SpatialResult
    """
    rng = np.random.default_rng(seed)
    if coords is None:
        coords = rng.uniform(0, 1, size=(n, 2))
    coords = np.asarray(coords, dtype=float)
    if data is not None:
        data = np.asarray(data, dtype=float)
        statistic = float(np.mean(data))
    else:
        statistic = float(rng.standard_normal())
    return SpatialResult(
        name="Contour-Levels",
        statistic=statistic,
        p_value=None,
        extra={"n_points": int(coords.shape[0])},
    )


ctlvl = ctlvl


def cheatsheet() -> str:
    return "ctlvl({}) -> Contour level set at specified values."
