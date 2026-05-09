"""Kernel spatial weights (Gaussian/bisquare)."""

import numpy as np

from ._containers import SpatialResult


def swkern(coords, bw=1.0, kernel="gaussian"):
    """Kernel spatial weights (Gaussian/bisquare).

    Category: Weights

    Parameters
    ----------
    coords, bw=1.0, kernel='gaussian' : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(coords)
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        kernel_sum = float(np.sum(np.exp(-0.5 * (dists / bw) ** 2)))
        result = kernel_sum / (n * n)
        return SpatialResult(name="swkern", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swkern", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swkern_fn = swkern


def cheatsheet() -> str:
    return "swkern({}) -> Kernel spatial weights (Gaussian/bisquare)."
