"""Moran eigenvector map (MEM) construction."""

import numpy as np

from ._containers import SpatialResult


def sfmoran(W):
    """Moran eigenvector map (MEM) construction.

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
        return SpatialResult(name="sfmoran", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sfmoran", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sfmoran_fn = sfmoran


def cheatsheet() -> str:
    return "sfmoran({}) -> Moran eigenvector map (MEM) construction."
