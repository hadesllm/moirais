"""MEM positive autocorrelation eigenvectors."""

import numpy as np

from ._containers import SpatialResult


def sfmemp(W):
    """MEM positive autocorrelation eigenvectors.

    Category: SFilter

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
        return SpatialResult(name="sfmemp", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sfmemp", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sfmemp_fn = sfmemp


def cheatsheet() -> str:
    return "sfmemp({}) -> MEM positive autocorrelation eigenvectors."
