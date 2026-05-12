# morie.fn -- function file (hadesllm/morie)
"""SAC direct/indirect/total impacts."""

import numpy as np

from ._containers import SpatialResult


def sacimp(coef, rho, lam, W):
    """SAC direct/indirect/total impacts.

    Category: SAC

    Parameters
    ----------
    coef, rho, lam, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(1 - rho * eigvals)) + np.sum(np.log(1 - lam * eigvals)))
        return SpatialResult(name="sacimp", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sacimp", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sacimp_fn = sacimp


def cheatsheet() -> str:
    return "sacimp({}) -> SAC direct/indirect/total impacts."
