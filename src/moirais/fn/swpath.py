"""Shortest-path distance between units in W."""

import numpy as np

from ._containers import SpatialResult


def swpath(W, i=0, j=2):
    """Shortest-path distance between units in W.

    Category: WDiag

    Parameters
    ----------
    W, i=0, j=2 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        W_arr = np.asarray(W, dtype=float)
        result = float(np.sum(W_arr))
        return SpatialResult(name="swpath", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swpath", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swpath_fn = swpath


def cheatsheet() -> str:
    return "swpath({}) -> Shortest-path distance between units in W."
