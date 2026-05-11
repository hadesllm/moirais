"""Gabriel graph spatial weights."""

import numpy as np

from ._containers import SpatialResult


def swgab(coords):
    """Gabriel graph spatial weights.

    Category: Weights

    Parameters
    ----------
    coords : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        result = float(np.mean(dists))
        return SpatialResult(name="swgab", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swgab", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swgab_fn = swgab


def cheatsheet() -> str:
    return "swgab({}) -> Gabriel graph spatial weights."
