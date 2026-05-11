"""Frobenius norm of W."""

import numpy as np

from ._containers import SpatialResult


def swfro(W):
    """Frobenius norm of W.

    Category: WDiag

    Parameters
    ----------
    W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        W_arr = np.asarray(W, dtype=float)
        result = float(np.sum(W_arr))
        return SpatialResult(name="swfro", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swfro", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swfro_fn = swfro


def cheatsheet() -> str:
    return "swfro({}) -> Frobenius norm of W."
