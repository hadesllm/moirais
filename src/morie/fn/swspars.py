"""Sparsify weights by threshold."""

import numpy as np

from ._containers import SpatialResult


def swspars(W, thr=0.1):
    """Sparsify weights by threshold.

    Category: Weights

    Parameters
    ----------
    W, thr=0.1 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        W_arr = np.asarray(W, dtype=float)
        result = float(np.sum(W_arr))
        return SpatialResult(name="swspars", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="swspars", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


swspars_fn = swspars


def cheatsheet() -> str:
    return "swspars({}) -> Sparsify weights by threshold."
