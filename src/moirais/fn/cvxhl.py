# moirais.fn — function file (hadesllm/moirais)
"""That which does not kill us makes us stronger. — Friedrich Nietzsche"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def convex_hull(points: np.ndarray) -> DescriptiveResult:
    """
    Compute the 2-D convex hull of a point set using the gift-wrapping
    (Jarvis march) algorithm.

    Time complexity is :math:`O(nh)` where *n* is the number of points
    and *h* is the number of hull vertices.

    :param points: (n, 2) array of 2-D points.
    :type points: numpy.ndarray
    :return: DescriptiveResult with hull vertex indices in extra.
    :rtype: DescriptiveResult
    :raises ValueError: If fewer than 3 points provided.

    References
    ----------
    Jarvis R.A. (1973). On the identification of the convex hull of a
    finite set of points in the plane. *Information Processing Letters*,
    2(1), 18-21.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be (n, 2) array.")
    n = len(pts)
    if n < 3:
        raise ValueError(f"Need >= 3 points for convex hull, got {n}.")
    start = int(np.argmin(pts[:, 0]))
    hull = []
    current = start
    while True:
        hull.append(current)
        candidate = 0
        for j in range(n):
            if j == current:
                continue
            cross = float(np.cross(pts[candidate] - pts[current], pts[j] - pts[current]))
            if candidate == current or cross < 0:
                candidate = j
            elif cross == 0:
                d_cand = np.sum((pts[candidate] - pts[current]) ** 2)
                d_j = np.sum((pts[j] - pts[current]) ** 2)
                if d_j > d_cand:
                    candidate = j
        current = candidate
        if current == start:
            break
    hull_arr = np.array(hull)
    return DescriptiveResult(
        name="convex_hull",
        value=len(hull),
        extra={"hull_indices": hull_arr, "hull_points": pts[hull_arr]},
    )


cvxhl = convex_hull


def cheatsheet() -> str:
    return "convex_hull({}) -> 2-D convex hull via gift wrapping. 'Impressive. Most impress"
