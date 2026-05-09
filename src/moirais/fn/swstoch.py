"""Test row-stochastic property of W."""

import numpy as np

from ._containers import SpatialResult


def swstoch(W):
    """Test row-stochastic property of W.

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
        return SpatialResult(name="swstoch", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="swstoch", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


swstoch_fn = swstoch


def cheatsheet() -> str:
    return "swstoch({}) -> Test row-stochastic property of W."
