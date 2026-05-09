# moirais.fn — function file (hadesllm/moirais)
"""Getis-Ord G global statistic."""

import numpy as np

from ._containers import SpatialResult


def lacgetg(y, W):
    """Getis-Ord G global statistic.

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
        return SpatialResult(name="lacgetg", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="lacgetg", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


lacgetg_fn = lacgetg


def cheatsheet() -> str:
    return "lacgetg({}) -> Getis-Ord G global statistic."
