# morie.fn — function file (hadesllm/morie)
"""Moran's I test on ML residuals."""

import numpy as np

from ._containers import SpatialResult


def miml(resid, W):
    """Moran's I test on ML residuals.

    Category: Moran

    Parameters
    ----------
    resid, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(resid)
        Wresid = np.dot(W, resid)
        result = float(np.dot(resid, Wresid) / (np.dot(resid, resid) + 1e-12))
        return SpatialResult(name="miml", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="miml", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


miml_fn = miml


def cheatsheet() -> str:
    return "miml({}) -> Moran's I test on ML residuals."
