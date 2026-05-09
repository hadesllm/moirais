# moirais.fn — function file (hadesllm/moirais)
"""MGWR cross-validation score."""

import numpy as np

from ._containers import SpatialResult


def mgwrcv(y, X, coords, bws=None):
    """MGWR cross-validation score.

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
        return SpatialResult(name="mgwrcv", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="mgwrcv", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


mgwrcv_fn = mgwrcv


def cheatsheet() -> str:
    return "mgwrcv({}) -> MGWR cross-validation score."
