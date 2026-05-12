# morie.fn -- function file (hadesllm/morie)
"""SAC variance-covariance matrix."""

import numpy as np

from ._containers import SpatialResult


def sacvar(X, W, rho, lam, sigma2):
    """SAC variance-covariance matrix.

    Category: SAC

    Parameters
    ----------
    X, W, rho, lam, sigma2 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(1 - rho * eigvals)) + np.sum(np.log(1 - lam * eigvals)))
        return SpatialResult(name="sacvar", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sacvar", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sacvar_fn = sacvar


def cheatsheet() -> str:
    return "sacvar({}) -> SAC variance-covariance matrix."
