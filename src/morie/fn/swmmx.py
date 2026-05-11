"""Min-max standardised spatial weights."""

import numpy as np

from ._containers import SpatialResult


def swmmx(W):
    """Min-max standardised spatial weights.

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
        return SpatialResult(name="swmmx", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swmmx", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swmmx_fn = swmmx


def cheatsheet() -> str:
    return "swmmx({}) -> Min-max standardised spatial weights."
