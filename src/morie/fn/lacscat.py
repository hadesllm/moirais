# morie.fn -- function file (hadesllm/morie)
"""Moran scatterplot quadrant classification."""

import numpy as np

from ._containers import SpatialResult


def lacscat(y, W):
    """Moran scatterplot quadrant classification.

    Category: Lattice

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
        return SpatialResult(name="lacscat", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="lacscat", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


lacscat_fn = lacscat


def cheatsheet() -> str:
    return "lacscat({}) -> Moran scatterplot quadrant classification."
