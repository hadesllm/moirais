# moirais.fn — function file (hadesllm/moirais)
"""SAC residual autocorrelation check."""

import numpy as np

from ._containers import SpatialResult


def sacres(resid, W):
    """SAC residual autocorrelation check.

    Category: SAC

    Parameters
    ----------
    resid, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(resid)
        Wresid = np.dot(W, resid)
        result = float(np.dot(resid, Wresid) / (np.dot(resid, resid) + 1e-12))
        return SpatialResult(name="sacres", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sacres", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sacres_fn = sacres


def cheatsheet() -> str:
    return "sacres({}) -> SAC residual autocorrelation check."
