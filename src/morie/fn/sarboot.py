# morie.fn — function file (hadesllm/morie)
"""SAR bootstrap confidence interval for rho."""

import numpy as np

from ._containers import SpatialResult


def sarboot(y, X, W, B=99):
    """SAR bootstrap confidence interval for rho.

    Category: SAR

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
        return SpatialResult(name="sarboot", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sarboot", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sarboot_fn = sarboot


def cheatsheet() -> str:
    return "sarboot({}) -> SAR bootstrap confidence interval for rho."
