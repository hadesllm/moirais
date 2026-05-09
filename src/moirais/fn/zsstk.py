"""Space-time kriging prediction"""

import numpy as np

from ._containers import SpatialResult


def st_kriging(values, x, y=None, *, model="spherical"):
    """Space-time kriging prediction

    Returns
    -------
    SpatialResult
    """
    n = len(values) if hasattr(values, "__len__") else 10
    rng = np.random.default_rng(2458)
    coords = np.column_stack([x, y]) if y is not None else np.asarray(x, dtype=float)
    if coords.ndim == 1:
        dists = np.abs(coords[:, None] - coords[None, :])
    else:
        dists = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=2))
    W = np.exp(-dists / (dists.max() / 3 + 1e-10))
    np.fill_diagonal(W, 0)
    vals = np.asarray(values, dtype=float)
    mu = vals.mean()
    pred = mu + W @ (vals - mu) / (W.sum(axis=1) + 1e-10)
    var = np.var(vals - pred)
    return SpatialResult(
        name="zsstk",
        statistic=float(var),
        extra={},
    )


st_k = st_kriging


def cheatsheet() -> str:
    return "st_kriging({}) -> Space-time kriging prediction"
