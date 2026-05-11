"""Spatial probit random-effects panel."""

import numpy as np

from ._containers import SpatialResult


def sptrx(y, X, W, unit_id):
    """Spatial probit random-effects panel.

    Category: SProbit

    Parameters
    ----------
    y : array-like
        Binary outcome (NT x 1).
    X : array-like
        Covariates (NT x k).
    W : array-like
        Spatial weights (n x n).
    unit_id : array-like
        Unit identifiers (NT x 1).

    Returns
    -------
    SpatialResult
    """
    try:
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        W = np.asarray(W, dtype=float)
        unit_id = np.asarray(unit_id)
        n = W.shape[0]
        unique_ids = np.unique(unit_id)
        y_means = np.array([float(np.mean(y[unit_id == uid])) for uid in unique_ids])
        stat = float(np.dot(y_means, np.dot(W, y_means)) / (np.dot(y_means, y_means) + 1e-12))
        return SpatialResult(name="sptrx", statistic=stat, p_value=None, extra={"n": n})
    except Exception:
        return SpatialResult(name="sptrx", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sptrx_fn = sptrx


def cheatsheet() -> str:
    return "sptrx({}) -> Spatial probit random-effects panel."
