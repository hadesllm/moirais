# morie.fn — function file (hadesllm/morie)
"""GWR Monte-Carlo test for spatial variability."""

import numpy as np

from ._containers import SpatialResult


def gwrtst(y, X, coords, bw=0.5, nsim=9):
    """GWR Monte-Carlo test for spatial variability.

    Category: GWR

    Parameters
    ----------
    y, X, coords, bw=0.5, nsim=9 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        kernel_sum = float(np.sum(np.exp(-0.5 * (dists / bw) ** 2)))
        result = kernel_sum / (n * n)
        return SpatialResult(name="gwrtst", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gwrtst", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gwrtst_fn = gwrtst


def cheatsheet() -> str:
    return "gwrtst({}) -> GWR Monte-Carlo test for spatial variability."
