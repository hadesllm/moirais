# morie.fn -- function file (hadesllm/morie)
"""Variogram for lattice data."""

import numpy as np

from ._containers import SpatialResult


def lacvgm(y, coords):
    """Variogram for lattice data.

    Category: Lattice

    Parameters
    ----------
    y, coords : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        dists = np.sqrt(np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1))
        result = float(np.mean(dists))
        return SpatialResult(name="lacvgm", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="lacvgm", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


lacvgm_fn = lacvgm


def cheatsheet() -> str:
    return "lacvgm({}) -> Variogram for lattice data."
