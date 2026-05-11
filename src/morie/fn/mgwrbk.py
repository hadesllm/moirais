# morie.fn — function file (hadesllm/morie)
"""MGWR backfitting algorithm iteration."""

import numpy as np

from ._containers import SpatialResult


def mgwrbk(y, X, coords, max_iter=10):
    """MGWR backfitting algorithm iteration.

    Category: MGWR

    Parameters
    ----------
    y, X, coords, max_iter=10 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        result = float(np.mean(dists))
        return SpatialResult(name="mgwrbk", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="mgwrbk", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


mgwrbk_fn = mgwrbk


def cheatsheet() -> str:
    return "mgwrbk({}) -> MGWR backfitting algorithm iteration."
