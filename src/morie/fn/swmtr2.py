"""Trace of W^2."""

import numpy as np

from ._containers import SpatialResult


def swmtr2(W):
    """Trace of W^2.

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
        return SpatialResult(name="swmtr2", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swmtr2", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swmtr2_fn = swmtr2


def cheatsheet() -> str:
    return "swmtr2({}) -> Trace of W^2."
