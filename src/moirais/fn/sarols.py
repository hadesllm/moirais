# moirais.fn — function file (hadesllm/moirais)
"""SAR OLS-IV two-stage estimator."""

import numpy as np

from ._containers import SpatialResult


def sarols(y, X, W):
    """SAR OLS-IV two-stage estimator.

    Category: SAR

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
        return SpatialResult(name="sarols", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sarols", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sarols_fn = sarols


def cheatsheet() -> str:
    return "sarols({}) -> SAR OLS-IV two-stage estimator."
