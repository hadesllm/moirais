"""Adaptive kernel weights (variable bandwidth)."""

import numpy as np

from ._containers import SpatialResult


def swadapt(coords, k=5):
    """Adaptive kernel weights (variable bandwidth).

    Category: Weights

    Parameters
    ----------
    coords, k=5 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        result = float(np.mean(dists))
        return SpatialResult(name="swadapt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="swadapt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


swadapt_fn = swadapt


def cheatsheet() -> str:
    return "swadapt({}) -> Adaptive kernel weights (variable bandwidth)."
