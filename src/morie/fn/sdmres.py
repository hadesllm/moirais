# morie.fn — function file (hadesllm/morie)
"""SDM residual autocorrelation check."""

import numpy as np

from ._containers import SpatialResult


def sdmres(resid, W):
    """SDM residual autocorrelation check.

    Category: SDM

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
        return SpatialResult(name="sdmres", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sdmres", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sdmres_fn = sdmres


def cheatsheet() -> str:
    return "sdmres({}) -> SDM residual autocorrelation check."
