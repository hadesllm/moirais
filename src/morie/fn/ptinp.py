# morie.fn -- function file (hadesllm/morie)
"""Statistics is the grammar of science. -- Karl Pearson"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> DescriptiveResult:
    """
    Determine whether a point lies inside a polygon using the ray-casting
    algorithm.

    Casts a horizontal ray from the test point to the right and counts
    edge crossings. An odd count means inside.

    :param point: (x, y) coordinates of the test point.
    :type point: tuple[float, float]
    :param polygon: (n, 2) array of polygon vertices in order.
    :type polygon: numpy.ndarray
    :return: DescriptiveResult with boolean inside flag.
    :rtype: DescriptiveResult

    References
    ----------
    Shimrat M. (1962). Algorithm 112: Position of point relative to
    polygon. *Communications of the ACM*, 5(8), 434.
    """
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("polygon must be (n, 2) array.")
    px, py = float(point[0]), float(point[1])
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return DescriptiveResult(
        name="point_in_polygon",
        value=inside,
        extra={"inside": inside, "point": point, "n_vertices": n},
    )


ptinp = point_in_polygon


def cheatsheet() -> str:
    return "point_in_polygon({}) -> Point-in-polygon test via ray casting. 'I find your lack of "
