"""SLX WX construction and summary."""

import numpy as np

from ._containers import SpatialResult


def slxwx(X, W):
    """SLX WX construction and summary.

    Category: SLX

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
        return SpatialResult(name="slxwx", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="slxwx", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


slxwx_fn = slxwx


def cheatsheet() -> str:
    return "slxwx({}) -> SLX WX construction and summary."
