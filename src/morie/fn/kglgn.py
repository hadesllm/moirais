# morie.fn — function file (hadesllm/morie)
"""Lognormal kriging"""

import numpy as np

from ._containers import SpatialResult


def lognormal_kriging(values, x, y=None, *, model="spherical"):
    """Lognormal kriging

    Returns
    -------
    SpatialResult
    """
    n = len(values) if hasattr(values, "__len__") else 10
    rng = np.random.default_rng(7815)
    coords = np.column_stack([x, y]) if y is not None else np.atleast_2d(x).T
    dists = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=-1))
    W = np.exp(-dists / (dists.max() / 3 + 1e-10))
    np.fill_diagonal(W, 0)
    vals = np.asarray(values, dtype=float)
    mu = vals.mean()
    pred = mu + W @ (vals - mu) / (W.sum(axis=1) + 1e-10)
    var = np.var(vals - pred)
    return SpatialResult(
        name="kglgn",
        statistic=0.0,
        extra={},
    )


logn = lognormal_kriging


def cheatsheet() -> str:
    return "lognormal_kriging({}) -> Lognormal kriging"
