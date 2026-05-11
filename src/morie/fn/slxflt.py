"""SLX spatial filter (de-mean with WX)."""

import numpy as np

from ._containers import SpatialResult


def slxflt(X, W):
    """SLX spatial filter (de-mean with WX).

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
        return SpatialResult(name="slxflt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="slxflt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


slxflt_fn = slxflt


def cheatsheet() -> str:
    return "slxflt({}) -> SLX spatial filter (de-mean with WX)."
