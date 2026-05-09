"""Nested / hierarchical spatial weights."""

import numpy as np

from ._containers import SpatialResult


def swnest(W_local, W_global):
    """Nested / hierarchical spatial weights.

    Category: Weights

    Parameters
    ----------
    W_local, W_global : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        W_combined = np.asarray(W_local, dtype=float) + np.asarray(W_global, dtype=float)
        result = float(np.sum(W_combined))
        return SpatialResult(name="swnest", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swnest", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swnest_fn = swnest


def cheatsheet() -> str:
    return "swnest({}) -> Nested / hierarchical spatial weights."
