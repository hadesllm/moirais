# morie.fn — function file (hadesllm/morie)
"""LISA (local indicators spatial association)."""

import numpy as np

from ._containers import SpatialResult


def laclisa(y, W):
    """LISA (local indicators spatial association).

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
        return SpatialResult(name="laclisa", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="laclisa", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


laclisa_fn = laclisa


def cheatsheet() -> str:
    return "laclisa({}) -> LISA (local indicators spatial association)."
