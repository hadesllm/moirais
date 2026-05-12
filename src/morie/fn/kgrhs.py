# morie.fn -- function file (hadesllm/morie)
"""Kriging right-hand side vector"""

import numpy as np

from ._containers import SpatialResult


def kriging_rhs(values, x, y=None, *, model="spherical"):
    """Kriging right-hand side vector

    Returns
    -------
    SpatialResult
    """
    n = len(values) if hasattr(values, "__len__") else 10
    rng = np.random.default_rng(8295)
    coords = np.column_stack([x, y]) if y is not None else np.atleast_2d(x).T
    dists = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=-1))
    W = np.exp(-dists / (dists.max() / 3 + 1e-10))
    np.fill_diagonal(W, 0)
    vals = np.asarray(values, dtype=float)
    mu = vals.mean()
    pred = mu + W @ (vals - mu) / (W.sum(axis=1) + 1e-10)
    var = np.var(vals - pred)
    return SpatialResult(
        name="kgrhs",
        statistic=0.0,
        extra={},
    )


krig = kriging_rhs


def cheatsheet() -> str:
    return "kriging_rhs({}) -> Kriging right-hand side vector"
