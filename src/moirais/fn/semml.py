# moirais.fn — function file (hadesllm/moirais)
"""SEM maximum-likelihood estimation."""

import numpy as np

from ._containers import SpatialResult


def semml(y, X, W):
    """SEM maximum-likelihood estimation.

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
        return SpatialResult(name="semml", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="semml", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


semml_fn = semml


def cheatsheet() -> str:
    return "semml({}) -> SEM maximum-likelihood estimation."
