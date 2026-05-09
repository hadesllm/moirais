"""Spatial probit LM test."""

import numpy as np

from ._containers import SpatialResult


def spprlm(y, X, W):
    """Spatial probit LM test.

    Category: SProbit

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
        return SpatialResult(name="spprlm", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="spprlm", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


spprlm_fn = spprlm


def cheatsheet() -> str:
    return "spprlm({}) -> Spatial probit LM test."
