"""Distance-band spatial weights matrix."""

import numpy as np

from ._containers import SpatialResult


def swdist(coords, d=1.0):
    """Distance-band spatial weights matrix.

    Category: Weights

    Parameters
    ----------
    coords, d=1.0 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        result = float(np.mean(dists))
        return SpatialResult(name="swdist", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swdist", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swdist_fn = swdist


def cheatsheet() -> str:
    return "swdist({}) -> Distance-band spatial weights matrix."
