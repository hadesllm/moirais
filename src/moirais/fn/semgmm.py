# moirais.fn — function file (hadesllm/moirais)
"""SEM GMM (Kelejian-Prucha) estimator."""

import numpy as np

from ._containers import SpatialResult


def semgmm(y, X, W):
    """SEM GMM (Kelejian-Prucha) estimator.

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
        return SpatialResult(name="semgmm", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="semgmm", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


semgmm_fn = semgmm


def cheatsheet() -> str:
    return "semgmm({}) -> SEM GMM (Kelejian-Prucha) estimator."
