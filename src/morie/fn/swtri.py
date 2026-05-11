"""Triangulate weights from Delaunay tessellation."""

import numpy as np

from ._containers import SpatialResult


def swtri(coords):
    """Triangulate weights from Delaunay tessellation.

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
        return SpatialResult(name="swtri", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swtri", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swtri_fn = swtri


def cheatsheet() -> str:
    return "swtri({}) -> Triangulate weights from Delaunay tessellation."
