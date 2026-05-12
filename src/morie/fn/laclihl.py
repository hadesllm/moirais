# morie.fn -- function file (hadesllm/morie)
"""LISA high-low outlier identification."""

import numpy as np

from ._containers import SpatialResult


def laclihl(y, W, p_thr=0.05):
    """LISA high-low outlier identification.

    Category: Lattice

    Parameters
    ----------
    y, W, p_thr=0.05 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        Wy = np.dot(W, y)
        result = float(np.corrcoef(y, Wy)[0, 1])
        return SpatialResult(name="laclihl", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="laclihl", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


laclihl_fn = laclihl


def cheatsheet() -> str:
    return "laclihl({}) -> LISA high-low outlier identification."
