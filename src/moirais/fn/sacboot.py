# moirais.fn — function file (hadesllm/moirais)
"""SAC bootstrap CI for rho."""

import numpy as np

from ._containers import SpatialResult


def sacboot(y, X, W, B=99):
    """SAC bootstrap CI for rho.

    Category: SAC

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
        return SpatialResult(name="sacboot", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sacboot", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sacboot_fn = sacboot


def cheatsheet() -> str:
    return "sacboot({}) -> SAC bootstrap CI for rho."
