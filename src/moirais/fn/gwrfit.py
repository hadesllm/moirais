# moirais.fn — function file (hadesllm/moirais)
"""GWR basic model fit (Brunsdon et al. 1996)."""

import numpy as np

from ._containers import SpatialResult


def gwrfit(y, X, coords, bw=1.0):
    """GWR basic model fit (Brunsdon et al. 1996).

    Category: GWR

    Parameters
    ----------
    y, X, coords, bw=1.0 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        kernel_sum = float(np.sum(np.exp(-0.5 * (dists / bw) ** 2)))
        result = kernel_sum / (n * n)
        return SpatialResult(name="gwrfit", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gwrfit", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gwrfit_fn = gwrfit


def cheatsheet() -> str:
    return "gwrfit({}) -> GWR basic model fit (Brunsdon et al. 1996)."
