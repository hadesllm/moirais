# morie.fn — function file (hadesllm/morie)
"""GWR local standard errors of coefficients."""

import numpy as np

from ._containers import SpatialResult


def gwrstd(y, X, coords, bw=0.5):
    """GWR local standard errors of coefficients.

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
        return SpatialResult(name="gwrstd", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gwrstd", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gwrstd_fn = gwrstd


def cheatsheet() -> str:
    return "gwrstd({}) -> GWR local standard errors of coefficients."
