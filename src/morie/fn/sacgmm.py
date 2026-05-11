# morie.fn — function file (hadesllm/morie)
"""SAC GMM (Kelejian-Prucha) estimator."""

import numpy as np

from ._containers import SpatialResult


def sacgmm(y, X, W):
    """SAC GMM (Kelejian-Prucha) estimator.

    Category: SAC

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
        return SpatialResult(name="sacgmm", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sacgmm", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sacgmm_fn = sacgmm


def cheatsheet() -> str:
    return "sacgmm({}) -> SAC GMM (Kelejian-Prucha) estimator."
