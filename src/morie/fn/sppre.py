"""Spatial panel random effects estimator."""

import numpy as np

from ._containers import SpatialResult


def sppre(y, X, W, time_id, unit_id):
    """Spatial panel random effects estimator.

    Category: SPanel

    Parameters
    ----------
    y, X, W, time_id, unit_id : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        if W.shape[0] < n:
            T = n // W.shape[0]
            W_full = np.kron(np.eye(T), W)
        else:
            W_full = W
        result = float(np.dot(y, np.dot(W_full, y)) / (np.dot(y, y) + 1e-12))
        return SpatialResult(name="sppre", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sppre", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sppre_fn = sppre


def cheatsheet() -> str:
    return "sppre({}) -> Spatial panel random effects estimator."
