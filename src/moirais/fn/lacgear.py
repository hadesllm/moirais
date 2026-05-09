# moirais.fn — function file (hadesllm/moirais)
"""Geary's C statistic."""

import numpy as np

from ._containers import SpatialResult


def lacgear(y, W):
    """Geary's C statistic.

    Category: Lattice

    Parameters
    ----------
    y, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        Wy = np.dot(W, y)
        result = float(np.corrcoef(y, Wy)[0, 1])
        return SpatialResult(name="lacgear", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="lacgear", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


lacgear_fn = lacgear


def cheatsheet() -> str:
    return "lacgear({}) -> Geary's C statistic."
