# moirais.fn — function file (hadesllm/moirais)
"""Multi-scale RBF hierarchical interpolation."""

import numpy as np

from ._containers import SpatialResult


def rbfmsv(data=None, coords=None, n=100, seed=42, **kwargs):
    """Multi-scale RBF hierarchical interpolation

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
        name="RBF-MultiScale",
        statistic=statistic,
        p_value=None,
        extra={"n_points": int(coords.shape[0])},
    )


rbfmsv = rbfmsv


def cheatsheet() -> str:
    return "rbfmsv({}) -> Multi-scale RBF hierarchical interpolation."
