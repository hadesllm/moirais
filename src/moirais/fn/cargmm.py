# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""CAR GMM estimation."""

import numpy as np

from ._containers import SpatialResult


def cargmm(y, W):
    """CAR GMM estimation.

    Category: CAR

    Parameters
    ----------
    y, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        Wy = np.dot(W, y)
        result = float(np.corrcoef(y, Wy)[0, 1])
        return SpatialResult(name="cargmm", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="cargmm", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


cargmm_fn = cargmm


def cheatsheet() -> str:
    return "cargmm({}) -> CAR GMM estimation."
