"""Inverse-distance spatial weights."""

import numpy as np

from ._containers import SpatialResult


def swinv(coords, power=1.0):
    """Inverse-distance spatial weights.

    Category: Weights

    Parameters
    ----------
    coords, power=1.0 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        result = float(np.mean(dists))
        return SpatialResult(name="swinv", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swinv", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swinv_fn = swinv


def cheatsheet() -> str:
    return "swinv({}) -> Inverse-distance spatial weights."
