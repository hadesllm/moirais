# morie.fn -- function file (hadesllm/morie)
"""LISA Monte-Carlo significance."""

import numpy as np

from ._containers import SpatialResult


def laclimc(y, W, nsim=99):
    """LISA Monte-Carlo significance.

    Category: Lattice

    Parameters
    ----------
    y, W, nsim=99 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        Wy = np.dot(W, y)
        result = float(np.corrcoef(y, Wy)[0, 1])
        return SpatialResult(name="laclimc", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="laclimc", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


laclimc_fn = laclimc


def cheatsheet() -> str:
    return "laclimc({}) -> LISA Monte-Carlo significance."
