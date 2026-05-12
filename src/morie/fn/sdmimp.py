# morie.fn -- function file (hadesllm/morie)
"""SDM direct/indirect/total impacts."""

import numpy as np

from ._containers import SpatialResult


def sdmimp(coef, theta, rho, W):
    """SDM direct/indirect/total impacts.

    Category: SDM

    Parameters
    ----------
    coef, theta, rho, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - rho * eigvals) + 1e-12)))
        return SpatialResult(name="sdmimp", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sdmimp", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sdmimp_fn = sdmimp


def cheatsheet() -> str:
    return "sdmimp({}) -> SDM direct/indirect/total impacts."
