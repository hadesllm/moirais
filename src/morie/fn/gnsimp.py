# morie.fn -- function file (hadesllm/morie)
"""GNS direct/indirect/total impacts."""

import numpy as np

from ._containers import SpatialResult


def gnsimp(coef, theta, rho, lam, W):
    """GNS direct/indirect/total impacts.

    Category: GNS

    Parameters
    ----------
    coef, theta, rho, lam, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(1 - rho * eigvals)) + np.sum(np.log(1 - lam * eigvals)))
        return SpatialResult(name="gnsimp", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gnsimp", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gnsimp_fn = gnsimp


def cheatsheet() -> str:
    return "gnsimp({}) -> GNS direct/indirect/total impacts."
