# morie.fn -- function file (hadesllm/morie)
"""Spatial NB marginal effects."""

import numpy as np

from ._containers import SpatialResult


def scnbmf(coef, rho, X, W):
    """Spatial NB marginal effects.

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
        return SpatialResult(name="scnbmf", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="scnbmf", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


scnbmf_fn = scnbmf


def cheatsheet() -> str:
    return "scnbmf({}) -> Spatial NB marginal effects."
