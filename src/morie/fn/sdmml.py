# morie.fn — function file (hadesllm/morie)
"""SDM maximum-likelihood estimation."""

import numpy as np

from ._containers import SpatialResult


def sdmml(y, X, W):
    """SDM maximum-likelihood estimation.

    Category: SDM

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
        return SpatialResult(name="sdmml", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sdmml", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sdmml_fn = sdmml


def cheatsheet() -> str:
    return "sdmml({}) -> SDM maximum-likelihood estimation."
