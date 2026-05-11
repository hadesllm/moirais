# morie.fn — function file (hadesllm/morie)
"""SEM residual autocorrelation (filtered residuals)."""

import numpy as np

from ._containers import SpatialResult


def semres(resid, W, lam=0.3):
    """SEM residual autocorrelation (filtered residuals).

    Category: SEM

    Parameters
    ----------
    resid, W, lam=0.3 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(resid)
        Wresid = np.dot(W, resid)
        result = float(np.dot(resid, Wresid) / (np.dot(resid, resid) + 1e-12))
        return SpatialResult(name="semres", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="semres", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


semres_fn = semres


def cheatsheet() -> str:
    return "semres({}) -> SEM residual autocorrelation (filtered residuals)."
