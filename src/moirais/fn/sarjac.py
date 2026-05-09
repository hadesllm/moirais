# moirais.fn — function file (hadesllm/moirais)
"""SAR Jacobian term for log-likelihood."""

import numpy as np

from ._containers import SpatialResult


def sarjac(W, rho):
    """SAR Jacobian term for log-likelihood.

    Category: SAR

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
        return SpatialResult(name="sarjac", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sarjac", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sarjac_fn = sarjac


def cheatsheet() -> str:
    return "sarjac({}) -> SAR Jacobian term for log-likelihood."
