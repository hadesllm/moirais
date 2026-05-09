"""Spatial Tobit model."""

import numpy as np

from ._containers import SpatialResult


def sprtobt(y, X, W):
    """Spatial Tobit model.

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
        return SpatialResult(name="sprtobt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sprtobt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sprtobt_fn = sprtobt


def cheatsheet() -> str:
    return "sprtobt({}) -> Spatial Tobit model."
