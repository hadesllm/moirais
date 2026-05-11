"""Second-order spatial lag W^2 y."""

import numpy as np

from ._containers import SpatialResult


def swlag2(W, y):
    """Second-order spatial lag W^2 y.

    Category: Weights

    Parameters
    ----------
    W, y : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        Wy = np.dot(W, y)
        result = float(np.corrcoef(y, Wy)[0, 1])
        return SpatialResult(name="swlag2", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swlag2", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swlag2_fn = swlag2


def cheatsheet() -> str:
    return "swlag2({}) -> Second-order spatial lag W^2 y."
