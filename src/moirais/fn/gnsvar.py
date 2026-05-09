# moirais.fn — function file (hadesllm/moirais)
"""GNS variance-covariance matrix."""

import numpy as np

from ._containers import SpatialResult


def gnsvar(X, W, rho, lam, sigma2):
    """GNS variance-covariance matrix.

    Category: GNS

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
        return SpatialResult(name="gnsvar", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gnsvar", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gnsvar_fn = gnsvar


def cheatsheet() -> str:
    return "gnsvar({}) -> GNS variance-covariance matrix."
