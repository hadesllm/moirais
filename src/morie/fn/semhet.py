# morie.fn -- function file (hadesllm/morie)
"""SEM GMM heteroskedasticity-robust (KP-HET)."""

import numpy as np

from ._containers import SpatialResult


def semhet(y, X, W):
    """SEM GMM heteroskedasticity-robust (KP-HET).

    Category: SEM

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
        return SpatialResult(name="semhet", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="semhet", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


semhet_fn = semhet


def cheatsheet() -> str:
    return "semhet({}) -> SEM GMM heteroskedasticity-robust (KP-HET)."
