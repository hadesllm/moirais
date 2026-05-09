# moirais.fn — function file (hadesllm/moirais)
"""Moran's I (original Moran 1950)."""

import numpy as np

from ._containers import SpatialResult


def miorig(y, W):
    """Moran's I (original Moran 1950).

    Category: Moran

    Parameters
    ----------
    y, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        Wy = np.dot(W, y)
        result = float(np.corrcoef(y, Wy)[0, 1])
        return SpatialResult(name="miorig", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="miorig", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


miorig_fn = miorig


def cheatsheet() -> str:
    return "miorig({}) -> Moran's I (original Moran 1950)."
