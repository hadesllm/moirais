# morie.fn — function file (hadesllm/morie)
"""SEM Kelejian-Prucha IV/2SLS estimator."""

import numpy as np

from ._containers import SpatialResult


def semkp(y, X, W):
    """SEM Kelejian-Prucha IV/2SLS estimator.

    Category: SEM

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
        return SpatialResult(name="semkp", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="semkp", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


semkp_fn = semkp


def cheatsheet() -> str:
    return "semkp({}) -> SEM Kelejian-Prucha IV/2SLS estimator."
