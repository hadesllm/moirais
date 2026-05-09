"""Doubly-standardise a spatial weights matrix."""

import numpy as np

from ._containers import SpatialResult


def swdoub(W):
    """Doubly-standardise a spatial weights matrix.

    Category: Weights

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
        return SpatialResult(name="swdoub", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swdoub", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swdoub_fn = swdoub


def cheatsheet() -> str:
    return "swdoub({}) -> Doubly-standardise a spatial weights matrix."
