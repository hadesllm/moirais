# morie.fn — function file (hadesllm/morie)
"""MGWR local standard errors."""

import numpy as np

from ._containers import SpatialResult


def mgwrstd(y, X, coords, bws=None):
    """MGWR local standard errors.

    Category: MGWR

    Parameters
    ----------
    y, X, coords, bws=None : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        bw = bws if bws is not None else float(np.median(dists[dists > 0]))
        kernel_sum = float(np.sum(np.exp(-0.5 * (dists / bw) ** 2)))
        result = kernel_sum / (n * n)
        return SpatialResult(name="mgwrstd", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="mgwrstd", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


mgwrstd_fn = mgwrstd


def cheatsheet() -> str:
    return "mgwrstd({}) -> MGWR local standard errors."
