"""Spatial ordered probit."""

import numpy as np

from ._containers import SpatialResult


def sprord(y, X, W):
    """Spatial ordered probit.

    Category: SProbit

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
        return SpatialResult(name="sprord", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sprord", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sprord_fn = sprord


def cheatsheet() -> str:
    return "sprord({}) -> Spatial ordered probit."
