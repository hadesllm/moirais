"""Power weights W^p."""

import numpy as np

from ._containers import SpatialResult


def swpower(W, p=2):
    """Power weights W^p.

    Category: Weights

    Parameters
    ----------
    W, p=2 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        W_arr = np.asarray(W, dtype=float)
        result = float(np.sum(W_arr))
        return SpatialResult(name="swpower", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="swpower", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


swpower_fn = swpower


def cheatsheet() -> str:
    return "swpower({}) -> Power weights W^p."
