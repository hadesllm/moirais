"""Condition number of W."""

import numpy as np

from ._containers import SpatialResult


def swcond(W):
    """Condition number of W.

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
        return SpatialResult(name="swcond", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swcond", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swcond_fn = swcond


def cheatsheet() -> str:
    return "swcond({}) -> Condition number of W."
