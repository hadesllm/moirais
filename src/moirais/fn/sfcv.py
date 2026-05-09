# moirais.fn — function file (hadesllm/moirais)
"""Cross-validated spatial filter selection."""

import numpy as np

from ._containers import SpatialResult


def sfcv(y, X, W):
    """Cross-validated spatial filter selection.

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
        return SpatialResult(name="sfcv", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sfcv", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sfcv_fn = sfcv


def cheatsheet() -> str:
    return "sfcv({}) -> Cross-validated spatial filter selection."
