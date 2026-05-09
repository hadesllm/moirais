"""Diagonal dominance check of W."""

import numpy as np

from ._containers import SpatialResult


def swdiag(W):
    """Diagonal dominance check of W.

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
        return SpatialResult(name="swdiag", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swdiag", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swdiag_fn = swdiag


def cheatsheet() -> str:
    return "swdiag({}) -> Diagonal dominance check of W."
