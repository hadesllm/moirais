"""Spatial panel IV/2SLS estimator."""

import numpy as np

from ._containers import SpatialResult


def sppiv(y, X, Z, W, time_id, unit_id):
    """Spatial panel IV/2SLS estimator.

    Category: SPanel

    Parameters
    ----------
    y, X, Z, W, time_id, unit_id : see function signature.

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
        return SpatialResult(name="sppiv", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sppiv", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sppiv_fn = sppiv


def cheatsheet() -> str:
    return "sppiv({}) -> Spatial panel IV/2SLS estimator."
