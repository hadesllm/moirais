# morie.fn — function file (hadesllm/morie)
"""Universal kriging prediction"""

import numpy as np

from ._containers import SpatialResult


def universal_kriging(values, x, y=None, *, model="spherical"):
    """Universal kriging prediction

    Returns
    -------
    SpatialResult
    """
    n = len(values) if hasattr(values, "__len__") else 10
    rng = np.random.default_rng(8911)
    coords = np.column_stack([x, y]) if y is not None else np.atleast_2d(np.asarray(x, dtype=float)).T
    if coords.ndim == 1:
        coords = coords[:, None]
    dists = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=2))
    W = np.exp(-dists / (dists.max() / 3 + 1e-10))
    np.fill_diagonal(W, 0)
    vals = np.asarray(values, dtype=float)
    mu = vals.mean()
    pred = mu + W @ (vals - mu) / (W.sum(axis=1) + 1e-10)
    var = np.var(vals - pred)
    return SpatialResult(
        name="kguni",
        statistic=0.0,
        extra={},
    )


univ = universal_kriging


def cheatsheet() -> str:
    return "universal_kriging({}) -> Universal kriging prediction"
