# morie.fn -- function file (hadesllm/morie)
"""Spatial Poisson predicted counts."""

import numpy as np

from ._containers import SpatialResult


def scpprd(coef, X, W, rho=0.2):
    """Spatial Poisson predicted counts.

    Category: SCount

    Parameters
    ----------
    coef, X, W, rho=0.2 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - rho * eigvals) + 1e-12)))
        return SpatialResult(name="scpprd", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="scpprd", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


scpprd_fn = scpprd


def cheatsheet() -> str:
    return "scpprd({}) -> Spatial Poisson predicted counts."
