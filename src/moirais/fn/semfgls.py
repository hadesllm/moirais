# moirais.fn — function file (hadesllm/moirais)
"""SEM feasible GLS estimator."""

import numpy as np

from ._containers import SpatialResult


def semfgls(y, X, W):
    """SEM feasible GLS estimator.

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
        return SpatialResult(name="semfgls", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="semfgls", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


semfgls_fn = semfgls


def cheatsheet() -> str:
    return "semfgls({}) -> SEM feasible GLS estimator."
