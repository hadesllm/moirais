# morie.fn — function file (hadesllm/morie)
"""LM test for SLX (spatially lagged X)."""

import numpy as np

from ._containers import SpatialResult


def lmslx(resid, X, W):
    """LM test for SLX (spatially lagged X).

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
        return SpatialResult(name="lmslx", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="lmslx", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


lmslx_fn = lmslx


def cheatsheet() -> str:
    return "lmslx({}) -> LM test for SLX (spatially lagged X)."
