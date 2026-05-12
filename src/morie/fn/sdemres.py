# morie.fn -- function file (hadesllm/morie)
"""SDEM residual autocorrelation check."""

import numpy as np

from ._containers import SpatialResult


def sdemres(resid, W):
    """SDEM residual autocorrelation check.

    Category: SDEM

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
        return SpatialResult(name="sdemres", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sdemres", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sdemres_fn = sdemres


def cheatsheet() -> str:
    return "sdemres({}) -> SDEM residual autocorrelation check."
