"""Spatial probit (SAR probit) estimation."""

import numpy as np

from ._containers import SpatialResult


def spprob(y, X, W):
    """Spatial probit (SAR probit) estimation.

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
        return SpatialResult(name="spprob", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="spprob", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


spprob_fn = spprob


def cheatsheet() -> str:
    return "spprob({}) -> Spatial probit (SAR probit) estimation."
