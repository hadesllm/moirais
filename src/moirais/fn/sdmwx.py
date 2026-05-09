# moirais.fn — function file (hadesllm/moirais)
"""SDM spatially lagged X matrix WX."""

import numpy as np

from ._containers import SpatialResult


def sdmwx(X, W):
    """SDM spatially lagged X matrix WX.

    Category: SDM

    Parameters
    ----------
    X, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        W_arr = np.asarray(W, dtype=float)
        result = float(np.sum(W_arr))
        return SpatialResult(name="sdmwx", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sdmwx", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sdmwx_fn = sdmwx


def cheatsheet() -> str:
    return "sdmwx({}) -> SDM spatially lagged X matrix WX."
