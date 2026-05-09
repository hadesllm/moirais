"""Spatial filter residual autocorrelation reduction."""

import numpy as np

from ._containers import SpatialResult


def sfredu(resid, W):
    """Spatial filter residual autocorrelation reduction.

    Category: SFilter

    Parameters
    ----------
    resid, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(resid)
        Wresid = np.dot(W, resid)
        result = float(np.dot(resid, Wresid) / (np.dot(resid, resid) + 1e-12))
        return SpatialResult(name="sfredu", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sfredu", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sfredu_fn = sfredu


def cheatsheet() -> str:
    return "sfredu({}) -> Spatial filter residual autocorrelation reduction."
