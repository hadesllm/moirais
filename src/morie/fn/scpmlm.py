# morie.fn — function file (hadesllm/morie)
"""Spatial Poisson LM test."""

import numpy as np

from ._containers import SpatialResult


def scpmlm(y, X, W):
    """Spatial Poisson LM test.

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
        return SpatialResult(name="scpmlm", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="scpmlm", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


scpmlm_fn = scpmlm


def cheatsheet() -> str:
    return "scpmlm({}) -> Spatial Poisson LM test."
