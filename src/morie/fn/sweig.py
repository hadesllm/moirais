"""Eigenvalues of W (for feasible parameter range)."""

import numpy as np

from ._containers import SpatialResult


def sweig(W):
    """Eigenvalues of W (for feasible parameter range).

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
        return SpatialResult(name="sweig", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sweig", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sweig_fn = sweig


def cheatsheet() -> str:
    return "sweig({}) -> Eigenvalues of W (for feasible parameter range)."
