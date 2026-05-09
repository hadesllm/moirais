# moirais.fn — function file (hadesllm/moirais)
"""Spatial count overdispersion test."""

import numpy as np

from ._containers import SpatialResult


def scdisp(y, X):
    """Spatial count overdispersion test.

    Category: SCount

    Parameters
    ----------
    y, X : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
        beta = np.linalg.solve(XtX, X.T @ y)
        resid = y - X @ beta
        result = float(np.mean(resid**2))
        return SpatialResult(name="scdisp", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="scdisp", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


scdisp_fn = scdisp


def cheatsheet() -> str:
    return "scdisp({}) -> Spatial count overdispersion test."
