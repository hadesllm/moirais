# moirais.fn — function file (hadesllm/moirais)
"""SAR residual spatial autocorrelation check."""

import numpy as np

from ._containers import SpatialResult


def sarres(resid, W):
    """SAR residual spatial autocorrelation check.

    Category: SAR

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
        return SpatialResult(name="sarres", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sarres", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sarres_fn = sarres


def cheatsheet() -> str:
    return "sarres({}) -> SAR residual spatial autocorrelation check."
