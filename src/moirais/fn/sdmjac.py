# moirais.fn — function file (hadesllm/moirais)
"""SDM Jacobian log-determinant."""

import numpy as np

from ._containers import SpatialResult


def sdmjac(W, rho):
    """SDM Jacobian log-determinant.

    Category: SDM

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
        return SpatialResult(name="sdmjac", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sdmjac", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sdmjac_fn = sdmjac


def cheatsheet() -> str:
    return "sdmjac({}) -> SDM Jacobian log-determinant."
