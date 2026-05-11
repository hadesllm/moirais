"""Spatial lag filter (I - rho*W)^-1."""

import numpy as np

from ._containers import SpatialResult


def swlagf(W, rho):
    """Spatial lag filter (I - rho*W)^-1.

    Category: WDiag

    Parameters
    ----------
    W, rho : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - rho * eigvals) + 1e-12)))
        return SpatialResult(name="swlagf", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swlagf", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swlagf_fn = swlagf


def cheatsheet() -> str:
    return "swlagf({}) -> Spatial lag filter (I - rho*W)^-1."
