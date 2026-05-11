"""Connectivity (avg neighbours) of weights."""

import numpy as np

from ._containers import SpatialResult


def swconn(W):
    """Connectivity (avg neighbours) of weights.

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
        return SpatialResult(name="swconn", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swconn", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swconn_fn = swconn


def cheatsheet() -> str:
    return "swconn({}) -> Connectivity (avg neighbours) of weights."
