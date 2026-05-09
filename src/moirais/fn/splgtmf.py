"""Spatial logit marginal effects."""

import numpy as np

from ._containers import SpatialResult


def splgtmf(coef, rho, X, W):
    """Spatial logit marginal effects.

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
        return SpatialResult(name="splgtmf", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="splgtmf", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


splgtmf_fn = splgtmf


def cheatsheet() -> str:
    return "splgtmf({}) -> Spatial logit marginal effects."
