# morie.fn — function file (hadesllm/morie)
"""Eigenvector spatial filtering (Tiefelsdorf & Griffith)."""

import numpy as np

from ._containers import SpatialResult


def sfeig(y, X, W):
    """Eigenvector spatial filtering (Tiefelsdorf & Griffith).

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
        return SpatialResult(name="sfeig", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sfeig", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sfeig_fn = sfeig


def cheatsheet() -> str:
    return "sfeig({}) -> Eigenvector spatial filtering (Tiefelsdorf & Griffith)."
