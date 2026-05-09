"""Spatial probit marginal effects."""

import numpy as np

from ._containers import SpatialResult


def spprmf(coef, rho, X, W):
    """Spatial probit marginal effects.

    Category: SProbit

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
        return SpatialResult(name="spprmf", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="spprmf", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


spprmf_fn = spprmf


def cheatsheet() -> str:
    return "spprmf({}) -> Spatial probit marginal effects."
