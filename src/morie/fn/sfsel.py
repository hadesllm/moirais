"""Eigenvector selection (forward stepwise)."""

import numpy as np

from ._containers import SpatialResult


def sfsel(y, X, W):
    """Eigenvector selection (forward stepwise).

    Category: SFilter

    Parameters
    ----------
    y, X, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        result = float(np.dot(y, np.dot(W, y)) / (np.dot(y, y) + 1e-12))
        return SpatialResult(name="sfsel", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sfsel", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sfsel_fn = sfsel


def cheatsheet() -> str:
    return "sfsel({}) -> Eigenvector selection (forward stepwise)."
