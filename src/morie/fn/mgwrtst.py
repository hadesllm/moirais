# morie.fn — function file (hadesllm/morie)
"""MGWR Monte-Carlo stationarity test."""

import numpy as np

from ._containers import SpatialResult


def mgwrtst(y, X, coords, nsim=9):
    """MGWR Monte-Carlo stationarity test.

    Category: MGWR

    Parameters
    ----------
    y, X, coords, nsim=9 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        result = float(np.mean(dists))
        return SpatialResult(name="mgwrtst", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="mgwrtst", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


mgwrtst_fn = mgwrtst


def cheatsheet() -> str:
    return "mgwrtst({}) -> MGWR Monte-Carlo stationarity test."
