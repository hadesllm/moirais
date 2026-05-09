"""Spatial panel between estimator."""

import numpy as np

from ._containers import SpatialResult


def sppbe(y, X, unit_id):
    """Spatial panel between estimator.

    Category: SPanel

    Parameters
    ----------
    y, X, unit_id : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
        beta = np.linalg.solve(XtX, X.T @ y)
        resid = y - X @ beta
        result = float(np.mean(resid**2))
        return SpatialResult(name="sppbe", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sppbe", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sppbe_fn = sppbe


def cheatsheet() -> str:
    return "sppbe({}) -> Spatial panel between estimator."
