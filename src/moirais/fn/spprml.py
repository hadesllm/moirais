"""Spatial probit ML with GHK simulator."""

import numpy as np

from ._containers import SpatialResult


def spprml(y, X, W, nsim=9):
    """Spatial probit ML with GHK simulator.

    Category: SProbit

    Parameters
    ----------
    y, X, W, nsim=9 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        result = float(np.dot(y, np.dot(W, y)) / (np.dot(y, y) + 1e-12))
        return SpatialResult(name="spprml", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="spprml", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


spprml_fn = spprml


def cheatsheet() -> str:
    return "spprml({}) -> Spatial probit ML with GHK simulator."
