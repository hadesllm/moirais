# morie.fn — function file (hadesllm/morie)
"""GWR leave-one-out cross-validation score."""

import numpy as np

from ._containers import SpatialResult


def gwrcv(y, X, coords, bw=0.5):
    """GWR leave-one-out cross-validation score.

    Category: GWR

    Parameters
    ----------
    y, X, coords, bw=0.5 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        kernel_sum = float(np.sum(np.exp(-0.5 * (dists / bw) ** 2)))
        result = kernel_sum / (n * n)
        return SpatialResult(name="gwrcv", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gwrcv", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gwrcv_fn = gwrcv


def cheatsheet() -> str:
    return "gwrcv({}) -> GWR leave-one-out cross-validation score."
