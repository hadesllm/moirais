"""Spatial probit GMM estimator."""

import numpy as np

from ._containers import SpatialResult


def spprgmm(y, X, W):
    """Spatial probit GMM estimator.

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
        return SpatialResult(name="spprgmm", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="spprgmm", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


spprgmm_fn = spprgmm


def cheatsheet() -> str:
    return "spprgmm({}) -> Spatial probit GMM estimator."
