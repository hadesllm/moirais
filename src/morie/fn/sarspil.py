# morie.fn — function file (hadesllm/morie)
"""SAR spillover ratio (indirect/direct)."""

import numpy as np

from ._containers import SpatialResult


def sarspil(coef, rho, W):
    """SAR spillover ratio (indirect/direct).

    Category: SAR

    Parameters
    ----------
    coef, rho, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - rho * eigvals) + 1e-12)))
        return SpatialResult(name="sarspil", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sarspil", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sarspil_fn = sarspil


def cheatsheet() -> str:
    return "sarspil({}) -> SAR spillover ratio (indirect/direct)."
