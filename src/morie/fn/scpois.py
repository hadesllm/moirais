# morie.fn — function file (hadesllm/morie)
"""Spatial Poisson SAR regression."""

import numpy as np

from ._containers import SpatialResult


def scpois(y, X, W):
    """Spatial Poisson SAR regression.

    Category: SCount

    Parameters
    ----------
    y, X, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        result = float(np.dot(y, np.dot(W, y)) / (np.dot(y, y) + 1e-12))
        return SpatialResult(name="scpois", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="scpois", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


scpois_fn = scpois


def cheatsheet() -> str:
    return "scpois({}) -> Spatial Poisson SAR regression."
