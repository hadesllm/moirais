# moirais.fn — function file (hadesllm/moirais)
"""Spatial zero-inflated negative binomial."""

import numpy as np

from ._containers import SpatialResult


def sczinb(y, X, W):
    """Spatial zero-inflated negative binomial.

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
        return SpatialResult(name="sczinb", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sczinb", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sczinb_fn = sczinb


def cheatsheet() -> str:
    return "sczinb({}) -> Spatial zero-inflated negative binomial."
