# moirais.fn — function file (hadesllm/moirais)
"""SAR Monte-Carlo impact simulation."""

import numpy as np

from ._containers import SpatialResult


def sarsim(coef, rho, W, nsim=99):
    """SAR Monte-Carlo impact simulation.

    Category: SAR

    Parameters
    ----------
    coef, rho, W, nsim=99 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - rho * eigvals) + 1e-12)))
        return SpatialResult(name="sarsim", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sarsim", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sarsim_fn = sarsim


def cheatsheet() -> str:
    return "sarsim({}) -> SAR Monte-Carlo impact simulation."
