"""Number of connected components in W."""

import numpy as np

from ._containers import SpatialResult


def swcomp(W):
    """Number of connected components in W.

    Category: WDiag

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
        return SpatialResult(name="swcomp", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swcomp", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swcomp_fn = swcomp


def cheatsheet() -> str:
    return "swcomp({}) -> Number of connected components in W."
