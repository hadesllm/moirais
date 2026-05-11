# morie.fn — function file (hadesllm/morie)
"""GNS (general nesting spatial) ML estimation."""

import numpy as np

from ._containers import SpatialResult


def gnsml(y, X, W):
    """GNS (general nesting spatial) ML estimation.

    Category: GNS

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
        return SpatialResult(name="gnsml", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gnsml", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gnsml_fn = gnsml


def cheatsheet() -> str:
    return "gnsml({}) -> GNS (general nesting spatial) ML estimation."
