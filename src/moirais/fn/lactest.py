# moirais.fn — function file (hadesllm/moirais)
"""General lattice spatial test battery."""

import numpy as np

from ._containers import SpatialResult


def lactest(y, W):
    """General lattice spatial test battery.

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
        return SpatialResult(name="lactest", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="lactest", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


lactest_fn = lactest


def cheatsheet() -> str:
    return "lactest({}) -> General lattice spatial test battery."
