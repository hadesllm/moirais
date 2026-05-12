# morie.fn -- function file (hadesllm/morie)
"""SAR variance-covariance of estimator."""

import numpy as np

from ._containers import SpatialResult


def sarvar(X, W, rho, sigma2):
    """SAR variance-covariance of estimator.

    Category: SAR

    Parameters
    ----------
    X, W, rho, sigma2 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - rho * eigvals) + 1e-12)))
        return SpatialResult(name="sarvar", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sarvar", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sarvar_fn = sarvar


def cheatsheet() -> str:
    return "sarvar({}) -> SAR variance-covariance of estimator."
