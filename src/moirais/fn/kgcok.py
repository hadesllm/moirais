# moirais.fn — function file (hadesllm/moirais)
"""Co-kriging multivariate"""

import numpy as np

from ._containers import SpatialResult


def cokriging(values, x, y=None, *, model="spherical"):
    """Co-kriging multivariate

    Returns
    -------
    SpatialResult
    """
    n = len(values) if hasattr(values, "__len__") else 10
    rng = np.random.default_rng(792)
    coords = np.column_stack([x, y]) if y is not None else np.asarray(x, dtype=float).reshape(-1, 1)
    dists = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=2))
    W = np.exp(-dists / (dists.max() / 3 + 1e-10))
    np.fill_diagonal(W, 0)
    vals = np.asarray(values, dtype=float)
    mu = vals.mean()
    pred = mu + W @ (vals - mu) / (W.sum(axis=1) + 1e-10)
    var = np.var(vals - pred)
    return SpatialResult(
        name="kgcok",
        statistic=0.0,
        extra={},
    )


cokr = cokriging


def cheatsheet() -> str:
    return "cokriging({}) -> Co-kriging multivariate"
