# morie.fn -- function file (hadesllm/morie)
"""Spatial Poisson marginal effects."""

import numpy as np

from ._containers import SpatialResult


def scpmf(coef, rho, X, W):
    """Spatial Poisson marginal effects.

    Category: SCount

    Parameters
    ----------
    coef, rho, X, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - rho * eigvals) + 1e-12)))
        return SpatialResult(name="scpmf", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="scpmf", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


scpmf_fn = scpmf


def cheatsheet() -> str:
    return "scpmf({}) -> Spatial Poisson marginal effects."
