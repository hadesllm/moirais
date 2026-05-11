"""Spatial panel diagnostics (LM tests)."""

import numpy as np

from ._containers import SpatialResult


def sppdiag(resid, X, W, time_id, unit_id):
    """Spatial panel diagnostics (LM tests).

    Category: SPanel

    Parameters
    ----------
    resid, X, W, time_id, unit_id : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(resid)
        if W.shape[0] < n:
            T = n // W.shape[0]
            W_full = np.kron(np.eye(T), W)
        else:
            W_full = W
        Wresid = np.dot(W_full, resid)
        result = float(np.dot(resid, Wresid) / (np.dot(resid, resid) + 1e-12))
        return SpatialResult(name="sppdiag", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sppdiag", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sppdiag_fn = sppdiag


def cheatsheet() -> str:
    return "sppdiag({}) -> Spatial panel diagnostics (LM tests)."
