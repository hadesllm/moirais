# morie.fn -- function file (hadesllm/morie)
"""Spatial correlation coefficient."""

import numpy as np

from ._containers import SpatialResult


def lacscor(y, W):
    """Spatial correlation coefficient.

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
        return SpatialResult(name="lacscor", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="lacscor", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


lacscor_fn = lacscor


def cheatsheet() -> str:
    return "lacscor({}) -> Spatial correlation coefficient."
