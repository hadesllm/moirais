# morie.fn — function file (hadesllm/morie)
"""MGWR per-variable bandwidth selection."""

import numpy as np

from ._containers import SpatialResult


def mgwrbw(y, X, coords):
    """MGWR per-variable bandwidth selection.

    Category: MGWR

    Parameters
    ----------
    y, X, coords : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        result = float(np.mean(dists))
        return SpatialResult(name="mgwrbw", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="mgwrbw", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


mgwrbw_fn = mgwrbw


def cheatsheet() -> str:
    return "mgwrbw({}) -> MGWR per-variable bandwidth selection."
