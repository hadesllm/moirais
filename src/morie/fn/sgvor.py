"""Voronoi tessellation for point patterns."""

from __future__ import annotations

from ._containers import DescriptiveResult


def voronoi_tessellation(points):
    """Compute Voronoi tessellation and cell areas for a point pattern.

    .. epigraph:: A journey of a thousand miles begins with a single step. -- Lao Tzu

    Parameters
    ----------
    points : array_like
        Point coordinates, shape ``(n, 2)``.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np
    from scipy.spatial import Voronoi

    pts = np.asarray(points, dtype=np.float64)
    vor = Voronoi(pts)

    areas = []
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            areas.append(np.inf)
        else:
            verts = vor.vertices[region]
            n_v = len(verts)
            area = 0.0
            for j in range(n_v):
                k = (j + 1) % n_v
                area += verts[j, 0] * verts[k, 1]
                area -= verts[k, 0] * verts[j, 1]
            areas.append(abs(area) / 2.0)

    finite_areas = [a for a in areas if np.isfinite(a)]
    cv = float(np.std(finite_areas) / np.mean(finite_areas)) if finite_areas and np.mean(finite_areas) > 0 else 0.0

    return DescriptiveResult(
        name="voronoi_tessellation",
        value=cv,
        extra={
            "cell_areas": areas,
            "vertices": [v.tolist() for v in vor.vertices],
            "n_finite_cells": len(finite_areas),
            "mean_area": float(np.mean(finite_areas)) if finite_areas else 0.0,
            "cv_area": cv,
        },
    )


sgvor = voronoi_tessellation


def cheatsheet() -> str:
    return "voronoi_tessellation({}) -> Voronoi tessellation for point patterns."
