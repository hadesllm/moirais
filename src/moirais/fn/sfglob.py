"""Global eigenvector spatial filter."""

import numpy as np

from ._containers import SpatialResult


def sfglob(y, X, W):
    """Global eigenvector spatial filter.

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
        return SpatialResult(name="sfglob", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sfglob", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sfglob_fn = sfglob


def cheatsheet() -> str:
    return "sfglob({}) -> Global eigenvector spatial filter."
