# morie.fn — function file (hadesllm/morie)
"""Join count test (BB, BW, WW)."""

import numpy as np

from ._containers import SpatialResult


def lacjoin(y_binary, W):
    """Join count test (BB, BW, WW).

    Category: Lattice

    Parameters
    ----------
    y_binary, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        y_arr = np.asarray(y_binary, dtype=float)
        Wy = np.dot(W, y_arr)
        result = float(np.dot(y_arr, Wy) / (np.dot(y_arr, y_arr) + 1e-12))
        return SpatialResult(name="lacjoin", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="lacjoin", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


lacjoin_fn = lacjoin


def cheatsheet() -> str:
    return "lacjoin({}) -> Join count test (BB, BW, WW)."
