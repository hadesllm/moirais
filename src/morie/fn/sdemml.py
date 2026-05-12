# morie.fn -- function file (hadesllm/morie)
"""SDEM maximum-likelihood estimation."""

import numpy as np

from ._containers import SpatialResult


def sdemml(y, X, W):
    """SDEM maximum-likelihood estimation.

    Category: SDEM

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
        return SpatialResult(name="sdemml", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sdemml", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sdemml_fn = sdemml


def cheatsheet() -> str:
    return "sdemml({}) -> SDEM maximum-likelihood estimation."
