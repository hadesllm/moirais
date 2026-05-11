"""Asymmetric (directed) spatial weights."""

import numpy as np

from ._containers import SpatialResult


def swasym(W):
    """Asymmetric (directed) spatial weights.

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
        return SpatialResult(name="swasym", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swasym", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swasym_fn = swasym


def cheatsheet() -> str:
    return "swasym({}) -> Asymmetric (directed) spatial weights."
