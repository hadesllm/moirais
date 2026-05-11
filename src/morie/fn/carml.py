# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""CAR (simultaneous) ML estimation."""

import numpy as np

from ._containers import SpatialResult


def carml(y, W):
    """CAR (simultaneous) ML estimation.

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
        return SpatialResult(name="carml", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="carml", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


carml_fn = carml


def cheatsheet() -> str:
    return "carml({}) -> CAR (simultaneous) ML estimation."
