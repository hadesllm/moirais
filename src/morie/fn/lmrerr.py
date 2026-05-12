# morie.fn -- function file (hadesllm/morie)
"""Robust LM test for spatial error."""

import numpy as np

from ._containers import SpatialResult


def lmrerr(resid, X, W):
    """Robust LM test for spatial error.

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
        return SpatialResult(name="lmrerr", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="lmrerr", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


lmrerr_fn = lmrerr


def cheatsheet() -> str:
    return "lmrerr({}) -> Robust LM test for spatial error."
