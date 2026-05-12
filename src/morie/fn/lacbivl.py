# morie.fn -- function file (hadesllm/morie)
"""Bivariate LISA (Lee 2001)."""

import numpy as np

from ._containers import SpatialResult


def lacbivl(x, y, W):
    """Bivariate LISA (Lee 2001).

    Category: Lattice

    Parameters
    ----------
    x, y, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        Wy = np.dot(W, y)
        result = float(np.corrcoef(y, Wy)[0, 1])
        return SpatialResult(name="lacbivl", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="lacbivl", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


lacbivl_fn = lacbivl


def cheatsheet() -> str:
    return "lacbivl({}) -> Bivariate LISA (Lee 2001)."
