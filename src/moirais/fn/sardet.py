# moirais.fn — function file (hadesllm/moirais)
"""SAR log-determinant  ln|I - rho*W|."""

import numpy as np

from ._containers import SpatialResult


def sardet(W, rho):
    """SAR log-determinant  ln|I - rho*W|.

    Category: SAR

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
        return SpatialResult(name="sardet", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sardet", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sardet_fn = sardet


def cheatsheet() -> str:
    return "sardet({}) -> SAR log-determinant  ln|I - rho*W|."
