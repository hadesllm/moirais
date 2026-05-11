# morie.fn — function file (hadesllm/morie)
"""SEM bootstrap CI for lambda."""

import numpy as np

from ._containers import SpatialResult


def semboot(y, X, W, B=99):
    """SEM bootstrap CI for lambda.

    Category: SEM

    Parameters
    ----------
    y, X, W, B=99 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        result = float(np.dot(y, np.dot(W, y)) / (np.dot(y, y) + 1e-12))
        return SpatialResult(name="semboot", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="semboot", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


semboot_fn = semboot


def cheatsheet() -> str:
    return "semboot({}) -> SEM bootstrap CI for lambda."
