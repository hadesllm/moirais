# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""CAR rho feasibility bounds."""

import numpy as np

from ._containers import SpatialResult


def carconv(W):
    """CAR rho feasibility bounds.

    Category: CAR

    Parameters
    ----------
    W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        W_arr = np.asarray(W, dtype=float)
        result = float(np.sum(W_arr))
        return SpatialResult(name="carconv", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="carconv", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


carconv_fn = carconv


def cheatsheet() -> str:
    return "carconv({}) -> CAR rho feasibility bounds."
