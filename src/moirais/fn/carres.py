# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""CAR residual Moran test."""

import numpy as np

from ._containers import SpatialResult


def carres(resid, W):
    """CAR residual Moran test.

    Category: CAR

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
        return SpatialResult(name="carres", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="carres", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


carres_fn = carres


def cheatsheet() -> str:
    return "carres({}) -> CAR residual Moran test."
