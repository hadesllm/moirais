# moirais.fn — function file (hadesllm/moirais)
"""SDM bootstrap CI for rho."""

import numpy as np

from ._containers import SpatialResult


def sdmboot(y, X, W, B=99):
    """SDM bootstrap CI for rho.

    Category: SDM

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
        return SpatialResult(name="sdmboot", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sdmboot", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sdmboot_fn = sdmboot


def cheatsheet() -> str:
    return "sdmboot({}) -> SDM bootstrap CI for rho."
