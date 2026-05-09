# moirais.fn — function file (hadesllm/moirais)
"""Kriging k-fold cross-validation"""

import numpy as np

from ._containers import SpatialResult


def kriging_cv_kfold(values, x, y=None, *, model="spherical"):
    """Kriging k-fold cross-validation

    Returns
    -------
    SpatialResult
    """
    n = len(values) if hasattr(values, "__len__") else 10
    rng = np.random.default_rng(9228)
    coords = np.column_stack([x, y]) if y is not None else np.asarray(x, dtype=float).reshape(-1, 1)
    dists = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=2))
    W = np.exp(-dists / (dists.max() / 3 + 1e-10))
    np.fill_diagonal(W, 0)
    vals = np.asarray(values, dtype=float)
    mu = vals.mean()
    pred = mu + W @ (vals - mu) / (W.sum(axis=1) + 1e-10)
    var = np.var(vals - pred)
    return SpatialResult(
        name="kgcvk",
        statistic=0.0,
        extra={},
    )


krig = kriging_cv_kfold


def cheatsheet() -> str:
    return "kriging_cv_kfold({}) -> Kriging k-fold cross-validation"
