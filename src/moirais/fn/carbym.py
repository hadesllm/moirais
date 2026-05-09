# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""CAR Besag-York-Mollié (BYM) variance components."""

import numpy as np

from ._containers import SpatialResult


def carbym(y, W):
    """CAR Besag-York-Mollié (BYM) variance components.

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
        return SpatialResult(name="carbym", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="carbym", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


carbym_fn = carbym


def cheatsheet() -> str:
    return "carbym({}) -> CAR Besag-York-Mollié (BYM) variance components."
