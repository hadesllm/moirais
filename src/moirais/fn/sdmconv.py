# moirais.fn — function file (hadesllm/moirais)
"""SDM rho/lambda joint feasibility check."""

import numpy as np

from ._containers import SpatialResult


def sdmconv(W, rho):
    """SDM rho/lambda joint feasibility check.

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
        return SpatialResult(name="sdmconv", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sdmconv", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sdmconv_fn = sdmconv


def cheatsheet() -> str:
    return "sdmconv({}) -> SDM rho/lambda joint feasibility check."
