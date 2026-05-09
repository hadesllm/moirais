# moirais.fn — function file (hadesllm/moirais)
"""GNS dual Jacobian ln|I-rho*W| + ln|I-lam*W|."""

import numpy as np

from ._containers import SpatialResult


def gnsjac(W, rho, lam):
    """GNS dual Jacobian ln|I-rho*W| + ln|I-lam*W|.

    Category: GNS

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
        return SpatialResult(name="gnsjac", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gnsjac", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gnsjac_fn = gnsjac


def cheatsheet() -> str:
    return "gnsjac({}) -> GNS dual Jacobian ln|I-rho*W| + ln|I-lam*W|."
