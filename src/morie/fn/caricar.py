# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Intrinsic CAR (ICAR) log-density."""

import numpy as np

from ._containers import SpatialResult


def caricar(phi, W):
    """Intrinsic CAR (ICAR) log-density.

    Category: CAR

    Parameters
    ----------
    phi, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        W_arr = np.asarray(W, dtype=float)
        result = float(np.sum(W_arr))
        return SpatialResult(name="caricar", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="caricar", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


caricar_fn = caricar


def cheatsheet() -> str:
    return "caricar({}) -> Intrinsic CAR (ICAR) log-density."
