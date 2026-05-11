# morie.fn — function file (hadesllm/morie)
"""Kelejian-Prucha LM test for SAC model."""

import numpy as np

from ._containers import SpatialResult


def lmkp(resid, X, W):
    """Kelejian-Prucha LM test for SAC model.

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
        return SpatialResult(name="lmkp", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="lmkp", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


lmkp_fn = lmkp


def cheatsheet() -> str:
    return "lmkp({}) -> Kelejian-Prucha LM test for SAC model."
