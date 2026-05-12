# morie.fn -- function file (hadesllm/morie)
"""Moran's I test on OLS residuals."""

import numpy as np

from ._containers import SpatialResult


def miols(resid, W):
    """Moran's I test on OLS residuals.

    Category: Moran

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
        return SpatialResult(name="miols", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="miols", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


miols_fn = miols


def cheatsheet() -> str:
    return "miols({}) -> Moran's I test on OLS residuals."
