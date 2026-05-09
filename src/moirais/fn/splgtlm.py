"""Spatial logit LM test."""

import numpy as np

from ._containers import SpatialResult


def splgtlm(y, X, W):
    """Spatial logit LM test.

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
        return SpatialResult(name="splgtlm", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="splgtlm", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


splgtlm_fn = splgtlm


def cheatsheet() -> str:
    return "splgtlm({}) -> Spatial logit LM test."
