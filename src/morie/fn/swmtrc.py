"""Trace of W (sum of diagonal)."""

import numpy as np

from ._containers import SpatialResult


def swmtrc(W):
    """Trace of W (sum of diagonal).

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
        return SpatialResult(name="swmtrc", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swmtrc", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swmtrc_fn = swmtrc


def cheatsheet() -> str:
    return "swmtrc({}) -> Trace of W (sum of diagonal)."
