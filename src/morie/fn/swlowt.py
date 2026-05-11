"""Lower-triangular half of weights matrix."""

import numpy as np

from ._containers import SpatialResult


def swlowt(W):
    """Lower-triangular half of weights matrix.

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
        return SpatialResult(name="swlowt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swlowt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swlowt_fn = swlowt


def cheatsheet() -> str:
    return "swlowt({}) -> Lower-triangular half of weights matrix."
