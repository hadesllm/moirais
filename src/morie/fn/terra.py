"""Terrain analysis. 'I was never on your side.' -- Terra"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def terrain_analysis(
    elevation: np.ndarray,
    *,
    cell_size: float = 1.0,
) -> DescriptiveResult:
    """Compute slope, aspect, and curvature from a digital elevation model.

    Uses 3x3 finite-difference kernels (Horn's method) to derive first and
    second derivative terrain metrics.

    Parameters
    ----------
    elevation : array (rows, cols)
        2-D elevation grid (DEM).
    cell_size : float
        Grid cell size in map units.

    Returns
    -------
    DescriptiveResult
        ``value`` = dict with ``'slope'``, ``'aspect'``, ``'curvature'`` grids.
    """
    Z = np.asarray(elevation, dtype=float)
    if Z.ndim != 2 or Z.shape[0] < 3 or Z.shape[1] < 3:
        raise ValueError("Elevation must be a 2-D array with at least 3x3 cells")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    padZ = np.pad(Z, 1, mode="edge")
    dz_dx = (
        (padZ[:-2, 2:] + 2 * padZ[1:-1, 2:] + padZ[2:, 2:]) - (padZ[:-2, :-2] + 2 * padZ[1:-1, :-2] + padZ[2:, :-2])
    ) / (8 * cell_size)
    dz_dy = (
        (padZ[2:, :-2] + 2 * padZ[2:, 1:-1] + padZ[2:, 2:]) - (padZ[:-2, :-2] + 2 * padZ[:-2, 1:-1] + padZ[:-2, 2:])
    ) / (8 * cell_size)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.mod(aspect_deg + 360, 360)
    d2z_dx2 = (padZ[1:-1, 2:] - 2 * padZ[1:-1, 1:-1] + padZ[1:-1, :-2]) / (cell_size**2)
    d2z_dy2 = (padZ[2:, 1:-1] - 2 * padZ[1:-1, 1:-1] + padZ[:-2, 1:-1]) / (cell_size**2)
    curvature = -(d2z_dx2 + d2z_dy2)
    return DescriptiveResult(
        name="Terrain analysis",
        value={
            "slope": slope_deg.tolist(),
            "aspect": aspect_deg.tolist(),
            "curvature": curvature.tolist(),
        },
        extra={
            "rows": Z.shape[0],
            "cols": Z.shape[1],
            "cell_size": cell_size,
            "mean_slope": float(np.mean(slope_deg)),
            "max_slope": float(np.max(slope_deg)),
            "mean_curvature": float(np.mean(curvature)),
        },
    )


terra = terrain_analysis


def cheatsheet() -> str:
    return "terrain_analysis({}) -> Terrain analysis. 'I was never on your side.' -- Terra"
