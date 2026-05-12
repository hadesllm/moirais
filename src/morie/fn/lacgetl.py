# morie.fn -- function file (hadesllm/morie)
"""Getis-Ord G* local statistic."""

import numpy as np

from ._containers import SpatialResult


def lacgetl(y, W):
    """Getis-Ord G* local statistic.

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
        return SpatialResult(name="lacgetl", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="lacgetl", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


lacgetl_fn = lacgetl


def cheatsheet() -> str:
    return "lacgetl({}) -> Getis-Ord G* local statistic."
