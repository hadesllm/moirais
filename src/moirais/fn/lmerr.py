# moirais.fn — function file (hadesllm/moirais)
"""LM test for spatial error (Anselin 1988)."""

import numpy as np

from ._containers import SpatialResult


def lmerr(resid, X, W):
    """LM test for spatial error (Anselin 1988).

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
        return SpatialResult(name="lmerr", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="lmerr", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


lmerr_fn = lmerr


def cheatsheet() -> str:
    return "lmerr({}) -> LM test for spatial error (Anselin 1988)."
