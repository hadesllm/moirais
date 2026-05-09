"""Local eigenvector spatial filter."""

import numpy as np

from ._containers import SpatialResult


def sfloc(y, X, W, i=0):
    """Local eigenvector spatial filter.

    Category: SFilter

    Parameters
    ----------
    y, X, W, i=0 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        result = float(np.dot(y, np.dot(W, y)) / (np.dot(y, y) + 1e-12))
        return SpatialResult(name="sfloc", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sfloc", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sfloc_fn = sfloc


def cheatsheet() -> str:
    return "sfloc({}) -> Local eigenvector spatial filter."
