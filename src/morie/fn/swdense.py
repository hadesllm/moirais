"""Compute density of spatial weights matrix."""

import numpy as np

from ._containers import SpatialResult


def swdense(W):
    """Compute density of spatial weights matrix.

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
        return SpatialResult(name="swdense", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="swdense", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


swdense_fn = swdense


def cheatsheet() -> str:
    return "swdense({}) -> Compute density of spatial weights matrix."
