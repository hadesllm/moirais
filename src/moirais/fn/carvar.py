# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""CAR variance-covariance matrix."""

import numpy as np

from ._containers import SpatialResult


def carvar(W, rho, sigma2):
    """CAR variance-covariance matrix.

    Category: CAR

    Parameters
    ----------
    W, rho, sigma2 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - rho * eigvals) + 1e-12)))
        return SpatialResult(name="carvar", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="carvar", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


carvar_fn = carvar


def cheatsheet() -> str:
    return "carvar({}) -> CAR variance-covariance matrix."
