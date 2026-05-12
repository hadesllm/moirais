# morie.fn -- function file (hadesllm/morie)
"""Robust LM test for spatial lag."""

import numpy as np

from ._containers import SpatialResult


def lmrlag(resid, X, W):
    """Robust LM test for spatial lag.

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
        return SpatialResult(name="lmrlag", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="lmrlag", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


lmrlag_fn = lmrlag


def cheatsheet() -> str:
    return "lmrlag({}) -> Robust LM test for spatial lag."
