# moirais.fn — function file (hadesllm/moirais)
"""MGWR hat matrix diagonal."""

import numpy as np

from ._containers import SpatialResult


def mgwrhat(y, X, coords, bws=None):
    """MGWR hat matrix diagonal.

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
        return SpatialResult(name="mgwrhat", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="mgwrhat", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


mgwrhat_fn = mgwrhat


def cheatsheet() -> str:
    return "mgwrhat({}) -> MGWR hat matrix diagonal."
