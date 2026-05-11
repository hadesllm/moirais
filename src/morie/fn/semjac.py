# morie.fn — function file (hadesllm/morie)
"""SEM Jacobian ln|I - lambda*W|."""

import numpy as np

from ._containers import SpatialResult


def semjac(W, lam):
    """SEM Jacobian ln|I - lambda*W|.

    Category: SEM

    Parameters
    ----------
    W, lam : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - lam * eigvals) + 1e-12)))
        return SpatialResult(name="semjac", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="semjac", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


semjac_fn = semjac


def cheatsheet() -> str:
    return "semjac({}) -> SEM Jacobian ln|I - lambda*W|."
