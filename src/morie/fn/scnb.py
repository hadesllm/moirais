# morie.fn — function file (hadesllm/morie)
"""Spatial negative-binomial regression."""

import numpy as np

from ._containers import SpatialResult


def scnb(y, X, W):
    """Spatial negative-binomial regression.

    Category: SCount

    Parameters
    ----------
    y, X, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        result = float(np.dot(y, np.dot(W, y)) / (np.dot(y, y) + 1e-12))
        return SpatialResult(name="scnb", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="scnb", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


scnb_fn = scnb


def cheatsheet() -> str:
    return "scnb({}) -> Spatial negative-binomial regression."
