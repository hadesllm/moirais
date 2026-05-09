"""Rank of spatial weights matrix."""

import numpy as np

from ._containers import SpatialResult


def swrank(W):
    """Rank of spatial weights matrix.

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
        return SpatialResult(name="swrank", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swrank", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swrank_fn = swrank


def cheatsheet() -> str:
    return "swrank({}) -> Rank of spatial weights matrix."
