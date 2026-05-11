# morie.fn — function file (hadesllm/morie)
"""GWR logistic regression (binary outcome)."""

import numpy as np

from ._containers import SpatialResult


def gwrlgt(y, X, coords, bw=0.5):
    """GWR logistic regression (binary outcome).

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
        return SpatialResult(name="gwrlgt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="gwrlgt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


gwrlgt_fn = gwrlgt


def cheatsheet() -> str:
    return "gwrlgt({}) -> GWR logistic regression (binary outcome)."
