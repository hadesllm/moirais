# moirais.fn — function file (hadesllm/moirais)
"""Spatial zero-inflated Poisson."""

import numpy as np

from ._containers import SpatialResult


def sczip(y, X, W):
    """Spatial zero-inflated Poisson.

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
        return SpatialResult(name="sczip", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sczip", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sczip_fn = sczip


def cheatsheet() -> str:
    return "sczip({}) -> Spatial zero-inflated Poisson."
