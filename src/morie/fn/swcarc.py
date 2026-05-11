"""Cardinality (neighbours count) for each unit."""

import numpy as np

from ._containers import SpatialResult


def swcarc(W):
    """Cardinality (neighbours count) for each unit.

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
        return SpatialResult(name="swcarc", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swcarc", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swcarc_fn = swcarc


def cheatsheet() -> str:
    return "swcarc({}) -> Cardinality (neighbours count) for each unit."
