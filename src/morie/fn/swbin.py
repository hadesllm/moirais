"""Binary (0/1) spatial weights."""

import numpy as np

from ._containers import SpatialResult


def swbin(W):
    """Binary (0/1) spatial weights.

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
        return SpatialResult(name="swbin", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swbin", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swbin_fn = swbin


def cheatsheet() -> str:
    return "swbin({}) -> Binary (0/1) spatial weights."
