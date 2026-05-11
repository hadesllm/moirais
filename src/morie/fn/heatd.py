# morie.fn — function file (hadesllm/morie)
"""2D kernel density for heatmaps. 'The belonging you seek is ahead.' -- Maz Kanata"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def heat_density(
    lat: np.ndarray,
    lon: np.ndarray,
    bandwidth: float = 0.01,
    grid_size: int = 100,
) -> DescriptiveResult:
    """2D Gaussian kernel density estimation for geographic heatmaps.

    Evaluates KDE on a regular grid over the bounding box of the
    input coordinates, producing a density surface suitable for
    heatmap visualization.

    Parameters
    ----------
    lat : ndarray, shape (n_points,)
        Latitudes (or y-coordinates).
    lon : ndarray, shape (n_points,)
        Longitudes (or x-coordinates).
    bandwidth : float
        Gaussian kernel bandwidth (in coordinate units).
    grid_size : int
        Number of grid points per axis.

    Returns
    -------
    DescriptiveResult
        name='Heat Density', value=peak density,
        extra has 'density' (grid_size x grid_size ndarray),
        'lat_grid', 'lon_grid', 'bandwidth'.

    References
    ----------
    Silverman, B.W. (1986). *Density Estimation for Statistics and
    Data Analysis*. Chapman & Hall. doi:10.1007/978-1-4899-3324-9
    """
    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon = np.asarray(lon, dtype=np.float64).ravel()
    n = len(lat)

    if n == 0:
        return DescriptiveResult(
            name="Heat Density",
            value=0.0,
            extra={
                "density": np.zeros((grid_size, grid_size)),
                "lat_grid": np.array([]),
                "lon_grid": np.array([]),
                "bandwidth": bandwidth,
            },
        )

    pad = bandwidth * 3
    lat_min, lat_max = lat.min() - pad, lat.max() + pad
    lon_min, lon_max = lon.min() - pad, lon.max() + pad

    lat_grid = np.linspace(lat_min, lat_max, grid_size)
    lon_grid = np.linspace(lon_min, lon_max, grid_size)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)

    density = np.zeros((grid_size, grid_size))
    h2 = 2.0 * bandwidth**2

    for i in range(n):
        dist_sq = (LAT - lat[i]) ** 2 + (LON - lon[i]) ** 2
        density += np.exp(-dist_sq / h2)

    density /= n * 2.0 * np.pi * bandwidth**2

    return DescriptiveResult(
        name="Heat Density",
        value=float(np.max(density)),
        extra={
            "density": density,
            "lat_grid": lat_grid,
            "lon_grid": lon_grid,
            "bandwidth": bandwidth,
            "grid_size": grid_size,
            "n_points": n,
        },
    )


heatd = heat_density


def cheatsheet() -> str:
    return "heat_density({}) -> 2D kernel density for heatmaps. 'The belonging you seek is a"
