# moirais.fn — function file (hadesllm/moirais)
"""The happiness of your life depends upon the quality of your thoughts. — Marcus Aurelius"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from ._containers import DescriptiveResult


def delaunay_simple(points: np.ndarray) -> DescriptiveResult:
    """
    Compute a Delaunay triangulation of 2-D points.

    Wraps scipy.spatial.Delaunay and returns simplices (triangle vertex
    indices) and the number of triangles.

    :param points: (n, 2) array of 2-D points. Need n >= 3.
    :type points: numpy.ndarray
    :return: DescriptiveResult with simplices and triangle count.
    :rtype: DescriptiveResult
    :raises ValueError: If fewer than 3 points or not 2-D.

    References
    ----------
    Delaunay B. (1934). Sur la sphere vide. *Bulletin de l'Academie des
    Sciences de l'URSS*, 6, 793-800.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be (n, 2) array.")
    if len(pts) < 3:
        raise ValueError(f"Need >= 3 points, got {len(pts)}.")
    tri = Delaunay(pts)
    return DescriptiveResult(
        name="delaunay_simple",
        value=len(tri.simplices),
        extra={"simplices": tri.simplices, "n_triangles": len(tri.simplices)},
    )


delny = delaunay_simple


def cheatsheet() -> str:
    return "The happiness of your life depends upon the quality of your thoughts. — Marcus Aurelius"
