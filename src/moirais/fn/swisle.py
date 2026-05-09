"""Identify islands (disconnected units) in W."""

import numpy as np

from ._containers import SpatialResult


def swisle(W):
    """Identify islands (disconnected units) in W.

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
        return SpatialResult(name="swisle", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swisle", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swisle_fn = swisle


def cheatsheet() -> str:
    return "swisle({}) -> Identify islands (disconnected units) in W."
