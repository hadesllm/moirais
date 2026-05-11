# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""CAR conditional simulation."""

import numpy as np

from ._containers import SpatialResult


def carsim(W, rho, sigma2, nsim=9):
    """CAR conditional simulation.

    Category: CAR

    Parameters
    ----------
    W, rho, sigma2, nsim=9 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - rho * eigvals) + 1e-12)))
        return SpatialResult(name="carsim", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="carsim", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


carsim_fn = carsim


def cheatsheet() -> str:
    return "carsim({}) -> CAR conditional simulation."
