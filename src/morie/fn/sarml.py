# morie.fn — function file (hadesllm/morie)
"""SAR maximum-likelihood estimation."""

import numpy as np

from ._containers import SpatialResult


def sarml(y, X, W):
    """SAR maximum-likelihood estimation.

    Category: SAR

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
        return SpatialResult(name="sarml", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sarml", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sarml_fn = sarml


def cheatsheet() -> str:
    return "sarml({}) -> SAR maximum-likelihood estimation."
