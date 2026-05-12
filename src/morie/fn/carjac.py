# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""CAR Jacobian term."""

import numpy as np

from ._containers import SpatialResult


def carjac(W, rho):
    """CAR Jacobian term.

    Category: CAR

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
        return SpatialResult(name="carjac", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="carjac", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


carjac_fn = carjac


def cheatsheet() -> str:
    return "carjac({}) -> CAR Jacobian term."
