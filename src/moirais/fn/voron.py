"""Voronoi tessellation areas. 'I have spoken.' -- Kuiil"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Voronoi

from ._containers import DescriptiveResult


def voronoi_areas(
    points: np.ndarray,
) -> DescriptiveResult:
    """Compute Voronoi tessellation and region areas.

    Constructs the Voronoi diagram for a set of 2D points and
    computes the area of each finite region.

    Parameters
    ----------
    points : ndarray, shape (n_points, 2)
        2D point coordinates.

    Returns
    -------
    DescriptiveResult
        name='Voronoi', value=mean finite area,
        extra has 'areas' (ndarray, NaN for infinite regions),
        'vertices', 'regions', 'n_finite', 'n_infinite'.

    References
    ----------
    Aurenhammer, F. (1991). Voronoi Diagrams -- A Survey of a
    Fundamental Geometric Data Structure. *ACM Computing Surveys*,
    23(3), 345-405. doi:10.1145/116873.116880
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be shape (n, 2)")

    n = points.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for Voronoi tessellation.")

    vor = Voronoi(points)

    areas = np.full(n, np.nan)
    n_finite = 0
    n_infinite = 0

    for i, reg_idx in enumerate(vor.point_region):
        region = vor.regions[reg_idx]
        if not region or -1 in region:
            n_infinite += 1
            continue
        verts = vor.vertices[region]
        area = _polygon_area(verts)
        areas[i] = area
        n_finite += 1

    finite_areas = areas[~np.isnan(areas)]
    mean_area = float(np.mean(finite_areas)) if len(finite_areas) > 0 else 0.0

    return DescriptiveResult(
        name="Voronoi",
        value=mean_area,
        extra={
            "areas": areas,
            "vertices": vor.vertices,
            "regions": vor.regions,
            "n_finite": n_finite,
            "n_infinite": n_infinite,
            "n_points": n,
        },
    )


def _polygon_area(verts: np.ndarray) -> float:
    """Shoelace formula for polygon area."""
    x = verts[:, 0]
    y = verts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


voron = voronoi_areas


def cheatsheet() -> str:
    return "voronoi_areas({}) -> Voronoi tessellation areas. 'I have spoken.' -- Kuiil"
