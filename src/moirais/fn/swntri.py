"""Number of non-zero entries (triples) in W."""

import numpy as np

from ._containers import SpatialResult


def swntri(W):
    """Number of non-zero entries (triples) in W.

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
        return SpatialResult(name="swntri", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swntri", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swntri_fn = swntri


def cheatsheet() -> str:
    return "swntri({}) -> Number of non-zero entries (triples) in W."
