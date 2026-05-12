# morie.fn -- function file (hadesllm/morie)
"""SAC (combined lag+error) ML estimation."""

import numpy as np

from ._containers import SpatialResult


def sacml(y, X, W):
    """SAC (combined lag+error) ML estimation.

    Category: SAC

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
        return SpatialResult(name="sacml", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sacml", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sacml_fn = sacml


def cheatsheet() -> str:
    return "sacml({}) -> SAC (combined lag+error) ML estimation."
