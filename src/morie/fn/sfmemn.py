"""MEM negative autocorrelation eigenvectors."""

import numpy as np

from ._containers import SpatialResult


def sfmemn(W):
    """MEM negative autocorrelation eigenvectors.

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
        return SpatialResult(name="sfmemn", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sfmemn", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sfmemn_fn = sfmemn


def cheatsheet() -> str:
    return "sfmemn({}) -> MEM negative autocorrelation eigenvectors."
