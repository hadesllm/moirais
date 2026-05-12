# morie.fn -- function file (hadesllm/morie)
"""SAC dual Jacobian term."""

import numpy as np

from ._containers import SpatialResult


def sacjac(W, rho, lam):
    """SAC dual Jacobian term.

    Category: SAC

    Parameters
    ----------
    W, rho, lam : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(1 - rho * eigvals)) + np.sum(np.log(1 - lam * eigvals)))
        return SpatialResult(name="sacjac", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sacjac", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sacjac_fn = sacjac


def cheatsheet() -> str:
    return "sacjac({}) -> SAC dual Jacobian term."
