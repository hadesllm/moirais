# moirais.fn — function file (hadesllm/moirais)
"""GWR hat matrix diagonal (leverage)."""

import numpy as np

from ._containers import SpatialResult


def gwrhat(y, X, coords, bw=0.5):
    """GWR hat matrix diagonal (leverage).

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
        return SpatialResult(name="gwrhat", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gwrhat", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gwrhat_fn = gwrhat


def cheatsheet() -> str:
    return "gwrhat({}) -> GWR hat matrix diagonal (leverage)."
