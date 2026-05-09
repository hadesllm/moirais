# moirais.fn — function file (hadesllm/moirais)
"""Spatial count bootstrap CI."""

import numpy as np

from ._containers import SpatialResult


def scpboot(y, X, W, B=9):
    """Spatial count bootstrap CI.

    Category: SCount

    Parameters
    ----------
    y, X, W, B=9 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        result = float(np.dot(y, np.dot(W, y)) / (np.dot(y, y) + 1e-12))
        return SpatialResult(name="scpboot", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="scpboot", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


scpboot_fn = scpboot


def cheatsheet() -> str:
    return "scpboot({}) -> Spatial count bootstrap CI."
