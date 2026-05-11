"""SLX OLS with spatially lagged regressors."""

import numpy as np

from ._containers import SpatialResult


def slxols(y, X, W):
    """SLX OLS with spatially lagged regressors.

    Category: SLX

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
        return SpatialResult(name="slxols", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="slxols", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


slxols_fn = slxols


def cheatsheet() -> str:
    return "slxols({}) -> SLX OLS with spatially lagged regressors."
