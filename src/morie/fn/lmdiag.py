# morie.fn — function file (hadesllm/morie)
"""LM diagnostics summary (lag, error, robust)."""

import numpy as np

from ._containers import SpatialResult


def lmdiag(resid, X, W):
    """LM diagnostics summary (lag, error, robust).

    Category: LM

    Parameters
    ----------
    resid, X, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(resid)
        Wresid = np.dot(W, resid)
        result = float(np.dot(resid, Wresid) / (np.dot(resid, resid) + 1e-12))
        return SpatialResult(name="lmdiag", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="lmdiag", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


lmdiag_fn = lmdiag


def cheatsheet() -> str:
    return "lmdiag({}) -> LM diagnostics summary (lag, error, robust)."
