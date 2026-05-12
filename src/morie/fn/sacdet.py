# morie.fn -- function file (hadesllm/morie)
"""SAC log-determinant product."""

import numpy as np

from ._containers import SpatialResult


def sacdet(W, rho, lam):
    """SAC log-determinant product.

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
        return SpatialResult(name="sacdet", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sacdet", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sacdet_fn = sacdet


def cheatsheet() -> str:
    return "sacdet({}) -> SAC log-determinant product."
