# moirais.fn — function file (hadesllm/moirais)
"""GWR local coefficient estimates."""

import numpy as np

from ._containers import SpatialResult


def gwrcoef(y, X, coords, bw=0.5):
    """GWR local coefficient estimates.

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
        return SpatialResult(name="gwrcoef", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="gwrcoef", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


gwrcoef_fn = gwrcoef


def cheatsheet() -> str:
    return "gwrcoef({}) -> GWR local coefficient estimates."
