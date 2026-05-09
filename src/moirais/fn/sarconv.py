# moirais.fn — function file (hadesllm/moirais)
"""SAR convergence diagnostic (eigenvalue method)."""

import numpy as np

from ._containers import SpatialResult


def sarconv(W):
    """SAR convergence diagnostic (eigenvalue method).

    Category: SAR

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
        return SpatialResult(name="sarconv", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sarconv", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sarconv_fn = sarconv


def cheatsheet() -> str:
    return "sarconv({}) -> SAR convergence diagnostic (eigenvalue method)."
