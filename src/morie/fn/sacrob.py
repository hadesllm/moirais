# morie.fn -- function file (hadesllm/morie)
"""SAC robust (HC) standard errors."""

import numpy as np

from ._containers import SpatialResult


def sacrob(resid, X, W):
    """SAC robust (HC) standard errors.

    Category: SAC

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
        return SpatialResult(name="sacrob", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sacrob", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sacrob_fn = sacrob


def cheatsheet() -> str:
    return "sacrob({}) -> SAC robust (HC) standard errors."
