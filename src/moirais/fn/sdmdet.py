# moirais.fn — function file (hadesllm/moirais)
"""SDM log-determinant ln|I - rho*W|."""

import numpy as np

from ._containers import SpatialResult


def sdmdet(W, rho):
    """SDM log-determinant ln|I - rho*W|.

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
        return SpatialResult(name="sdmdet", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sdmdet", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sdmdet_fn = sdmdet


def cheatsheet() -> str:
    return "sdmdet({}) -> SDM log-determinant ln|I - rho*W|."
