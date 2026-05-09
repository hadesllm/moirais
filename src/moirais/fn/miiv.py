# moirais.fn — function file (hadesllm/moirais)
"""Moran's I test on IV residuals."""

import numpy as np

from ._containers import SpatialResult


def miiv(resid, W):
    """Moran's I test on IV residuals.

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
        return SpatialResult(name="miiv", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="miiv", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


miiv_fn = miiv


def cheatsheet() -> str:
    return "miiv({}) -> Moran's I test on IV residuals."
