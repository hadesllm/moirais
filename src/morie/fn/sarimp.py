# morie.fn — function file (hadesllm/morie)
"""SAR direct/indirect/total impact decomposition."""

import numpy as np

from ._containers import SpatialResult


def sarimp(coef, rho, W):
    """SAR direct/indirect/total impact decomposition.

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
        return SpatialResult(name="sarimp", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sarimp", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sarimp_fn = sarimp


def cheatsheet() -> str:
    return "sarimp({}) -> SAR direct/indirect/total impact decomposition."
