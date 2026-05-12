# morie.fn -- function file (hadesllm/morie)
"""SAR GMM estimator."""

import numpy as np

from ._containers import SpatialResult


def sargmm(y, X, W):
    """SAR GMM estimator.

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
        return SpatialResult(name="sargmm", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sargmm", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sargmm_fn = sargmm


def cheatsheet() -> str:
    return "sargmm({}) -> SAR GMM estimator."
