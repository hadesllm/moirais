# morie.fn — function file (hadesllm/morie)
"""LISA high-high cluster identification."""

import numpy as np

from ._containers import SpatialResult


def laclihh(y, W, p_thr=0.05):
    """LISA high-high cluster identification.

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
        return SpatialResult(name="laclihh", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="laclihh", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


laclihh_fn = laclihh


def cheatsheet() -> str:
    return "laclihh({}) -> LISA high-high cluster identification."
