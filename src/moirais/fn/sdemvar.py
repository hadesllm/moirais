# moirais.fn — function file (hadesllm/moirais)
"""SDEM variance-covariance matrix."""

import numpy as np

from ._containers import SpatialResult


def sdemvar(X, WX, W, lam, sigma2):
    """SDEM variance-covariance matrix.

    Category: SDEM

    Parameters
    ----------
    X, WX, W, lam, sigma2 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - lam * eigvals) + 1e-12)))
        return SpatialResult(name="sdemvar", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sdemvar", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sdemvar_fn = sdemvar


def cheatsheet() -> str:
    return "sdemvar({}) -> SDEM variance-covariance matrix."
