# morie.fn -- function file (hadesllm/morie)
"""GWR seemingly-unrelated regression system."""

import numpy as np

from ._containers import SpatialResult


def gwrsur(ys, X, coords, bw=0.5):
    """GWR seemingly-unrelated regression system.

    Category: GWR

    Parameters
    ----------
    ys, X, coords, bw=0.5 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(ys[0])
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        kernel_sum = float(np.sum(np.exp(-0.5 * (dists / bw) ** 2)))
        result = kernel_sum / (n * n)
        return SpatialResult(name="gwrsur", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gwrsur", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gwrsur_fn = gwrsur


def cheatsheet() -> str:
    return "gwrsur({}) -> GWR seemingly-unrelated regression system."
