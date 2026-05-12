# morie.fn -- function file (hadesllm/morie)
"""SDM variance-covariance matrix."""

import numpy as np

from ._containers import SpatialResult


def sdmvar(X, WX, W, rho, sigma2):
    """SDM variance-covariance matrix.

    Category: SDM

    Parameters
    ----------
    X, WX, W, rho, sigma2 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - rho * eigvals) + 1e-12)))
        return SpatialResult(name="sdmvar", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sdmvar", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sdmvar_fn = sdmvar


def cheatsheet() -> str:
    return "sdmvar({}) -> SDM variance-covariance matrix."
