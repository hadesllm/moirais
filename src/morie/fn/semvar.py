# morie.fn — function file (hadesllm/morie)
"""SEM variance-covariance matrix of estimator."""

import numpy as np

from ._containers import SpatialResult


def semvar(X, W, lam, sigma2):
    """SEM variance-covariance matrix of estimator.

    Category: SEM

    Parameters
    ----------
    X, W, lam, sigma2 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - lam * eigvals) + 1e-12)))
        return SpatialResult(name="semvar", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="semvar", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


semvar_fn = semvar


def cheatsheet() -> str:
    return "semvar({}) -> SEM variance-covariance matrix of estimator."
