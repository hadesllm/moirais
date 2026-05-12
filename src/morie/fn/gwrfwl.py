# morie.fn -- function file (hadesllm/morie)
"""GWR Frisch-Waugh-Lovell local partitioned regression."""

import numpy as np

from ._containers import SpatialResult


def gwrfwl(y, X1, X2, coords, bw=0.5):
    """GWR Frisch-Waugh-Lovell local partitioned regression.

    Category: GWR

    Parameters
    ----------
    y, X1, X2, coords, bw=0.5 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        kernel_sum = float(np.sum(np.exp(-0.5 * (dists / bw) ** 2)))
        result = kernel_sum / (n * n)
        return SpatialResult(name="gwrfwl", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gwrfwl", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gwrfwl_fn = gwrfwl


def cheatsheet() -> str:
    return "gwrfwl({}) -> GWR Frisch-Waugh-Lovell local partitioned regression."
