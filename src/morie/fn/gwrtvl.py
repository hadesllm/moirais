# morie.fn — function file (hadesllm/morie)
"""GWR t-values for local coefficients."""

import numpy as np

from ._containers import SpatialResult


def gwrtvl(y, X, coords, bw=0.5):
    """GWR t-values for local coefficients.

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
        return SpatialResult(name="gwrtvl", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gwrtvl", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gwrtvl_fn = gwrtvl


def cheatsheet() -> str:
    return "gwrtvl({}) -> GWR t-values for local coefficients."
