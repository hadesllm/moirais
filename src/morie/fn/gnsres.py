# morie.fn — function file (hadesllm/morie)
"""GNS residual autocorrelation check."""

import numpy as np

from ._containers import SpatialResult


def gnsres(resid, W):
    """GNS residual autocorrelation check.

    Category: GNS

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
        return SpatialResult(name="gnsres", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gnsres", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gnsres_fn = gnsres


def cheatsheet() -> str:
    return "gnsres({}) -> GNS residual autocorrelation check."
