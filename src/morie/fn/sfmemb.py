"""MEM Bonferroni-corrected eigenvector selection."""

import numpy as np

from ._containers import SpatialResult


def sfmemb(y, W, alpha=0.05):
    """MEM Bonferroni-corrected eigenvector selection.

    Category: SFilter

    Parameters
    ----------
    y, W, alpha=0.05 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        Wy = np.dot(W, y)
        result = float(np.corrcoef(y, Wy)[0, 1])
        return SpatialResult(name="sfmemb", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sfmemb", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sfmemb_fn = sfmemb


def cheatsheet() -> str:
    return "sfmemb({}) -> MEM Bonferroni-corrected eigenvector selection."
