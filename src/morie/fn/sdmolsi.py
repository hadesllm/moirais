# morie.fn -- function file (hadesllm/morie)
"""SDM OLS ignoring spatial component (baseline)."""

import numpy as np

from ._containers import SpatialResult


def sdmolsi(y, X):
    """SDM OLS ignoring spatial component (baseline).

    Category: SDM

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
        return SpatialResult(name="sdmolsi", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sdmolsi", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sdmolsi_fn = sdmolsi


def cheatsheet() -> str:
    return "sdmolsi({}) -> SDM OLS ignoring spatial component (baseline)."
