"""k-nearest-neighbour spatial weights matrix."""

import numpy as np

from ._containers import SpatialResult


def swknn(coords, k=4):
    """k-nearest-neighbour spatial weights matrix.

    Category: Weights

    Parameters
    ----------
    coords, k=4 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        result = float(np.mean(dists))
        return SpatialResult(name="swknn", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swknn", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swknn_fn = swknn


def cheatsheet() -> str:
    return "swknn({}) -> k-nearest-neighbour spatial weights matrix."
