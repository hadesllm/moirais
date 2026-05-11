"""Spatial lag Wy."""

import numpy as np

from ._containers import SpatialResult


def swlag(W, y):
    """Spatial lag Wy.

    Category: Weights

    Parameters
    ----------
    W, y : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        Wy = np.dot(W, y)
        result = float(np.corrcoef(y, Wy)[0, 1])
        return SpatialResult(name="swlag", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swlag", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swlag_fn = swlag


def cheatsheet() -> str:
    return "swlag({}) -> Spatial lag Wy."
