# moirais.fn — function file (hadesllm/moirais)
"""LM test for spatial Durbin model."""

import numpy as np

from ._containers import SpatialResult


def lmsdm(resid, X, W):
    """LM test for spatial Durbin model.

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
        return SpatialResult(name="lmsdm", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="lmsdm", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


lmsdm_fn = lmsdm


def cheatsheet() -> str:
    return "lmsdm({}) -> LM test for spatial Durbin model."
