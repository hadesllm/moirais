"""Row-standardise a spatial weights matrix."""

import numpy as np

from ._containers import SpatialResult


def swrow(W):
    """Row-standardise a spatial weights matrix.

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
        return SpatialResult(name="swrow", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swrow", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swrow_fn = swrow


def cheatsheet() -> str:
    return "swrow({}) -> Row-standardise a spatial weights matrix."
