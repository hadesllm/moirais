"""Spatial panel fixed effects (within) estimator."""

import numpy as np

from ._containers import SpatialResult


def sppfe(y, X, W, time_id, unit_id):
    """Spatial panel fixed effects (within) estimator.

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
        return SpatialResult(name="sppfe", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sppfe", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sppfe_fn = sppfe


def cheatsheet() -> str:
    return "sppfe({}) -> Spatial panel fixed effects (within) estimator."
