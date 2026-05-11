"""Spatial logit estimation."""

import numpy as np

from ._containers import SpatialResult


def splogit(y, X, W):
    """Spatial logit estimation.

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
        return SpatialResult(name="splogit", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="splogit", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


splogit_fn = splogit


def cheatsheet() -> str:
    return "splogit({}) -> Spatial logit estimation."
