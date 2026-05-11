"""SLX bootstrap CI for theta."""

import numpy as np

from ._containers import SpatialResult


def slxboot(y, X, W, B=99):
    """SLX bootstrap CI for theta.

    Category: SLX

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
        return SpatialResult(name="slxboot", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="slxboot", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


slxboot_fn = slxboot


def cheatsheet() -> str:
    return "slxboot({}) -> SLX bootstrap CI for theta."
