# morie.fn -- function file (hadesllm/morie)
"""SAC rho/lambda joint feasibility bounds."""

import numpy as np

from ._containers import SpatialResult


def sacconv(W, rho, lam):
    """SAC rho/lambda joint feasibility bounds.

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
        return SpatialResult(name="sacconv", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sacconv", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sacconv_fn = sacconv


def cheatsheet() -> str:
    return "sacconv({}) -> SAC rho/lambda joint feasibility bounds."
