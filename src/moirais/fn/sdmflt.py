# moirais.fn — function file (hadesllm/moirais)
"""SDM spatial filter transform."""

import numpy as np

from ._containers import SpatialResult


def sdmflt(y, W, rho=0.3):
    """SDM spatial filter transform.

    Category: SDM

    Parameters
    ----------
    y, W, rho=0.3 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        Wy = np.dot(W, y)
        result = float(np.corrcoef(y, Wy)[0, 1])
        return SpatialResult(name="sdmflt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sdmflt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sdmflt_fn = sdmflt


def cheatsheet() -> str:
    return "sdmflt({}) -> SDM spatial filter transform."
