# morie.fn -- function file (hadesllm/morie)
"""LM test for spatial lag (Anselin 1988)."""

import numpy as np

from ._containers import SpatialResult


def lmlag(resid, X, W):
    """LM test for spatial lag (Anselin 1988).

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
        return SpatialResult(name="lmlag", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="lmlag", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


lmlag_fn = lmlag


def cheatsheet() -> str:
    return "lmlag({}) -> LM test for spatial lag (Anselin 1988)."
