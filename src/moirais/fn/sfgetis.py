"""Getis spatial filtering approach."""

import numpy as np

from ._containers import SpatialResult


def sfgetis(y, W):
    """Getis spatial filtering approach.

    Category: SFilter

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
        return SpatialResult(name="sfgetis", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sfgetis", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sfgetis_fn = sfgetis


def cheatsheet() -> str:
    return "sfgetis({}) -> Getis spatial filtering approach."
