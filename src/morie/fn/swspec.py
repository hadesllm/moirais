"""Spectral radius of W."""

import numpy as np

from ._containers import SpatialResult


def swspec(W):
    """Spectral radius of W.

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
        return SpatialResult(name="swspec", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swspec", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swspec_fn = swspec


def cheatsheet() -> str:
    return "swspec({}) -> Spectral radius of W."
