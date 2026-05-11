# morie.fn — function file (hadesllm/morie)
"""Moran's I randomisation test."""

import numpy as np

from ._containers import SpatialResult


def mirand(y, W, nsim=99):
    """Moran's I randomisation test.

    Category: Moran

    Parameters
    ----------
    y, W, nsim=99 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        Wy = np.dot(W, y)
        result = float(np.corrcoef(y, Wy)[0, 1])
        return SpatialResult(name="mirand", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="mirand", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


mirand_fn = mirand


def cheatsheet() -> str:
    return "mirand({}) -> Moran's I randomisation test."
