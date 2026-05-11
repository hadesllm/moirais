# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Out of chaos, comes order. — Friedrich Nietzsche"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _validate_df


def idw_interpolate(
    data: pd.DataFrame,
    *,
    x: str = "x",
    y: str = "y",
    z: str = "z",
    grid_size: int = 20,
    power: float = 2.0,
) -> DescriptiveResult:
    """Inverse distance weighted (IDW) interpolation on a regular grid.

    Given scattered observations of a spatial variable, constructs a regular
    grid and estimates values using IDW.  Commonly used in hydrogeology for
    water table elevation interpolation.

    Parameters
    ----------
    data : DataFrame
        Must contain x-coordinate, y-coordinate, and z-value columns.
    x, y, z : str
        Column names.
    grid_size : int
        Number of grid points per axis.
    power : float
        Distance weighting exponent (higher = more local).

    Returns
    -------
    DescriptiveResult
        ``value`` = interpolated grid as a 2D list.
    """
    _validate_df(data, x, y, z)
    df = data[[x, y, z]].dropna()
    xs = df[x].to_numpy(dtype=float)
    ys = df[y].to_numpy(dtype=float)
    zs = df[z].to_numpy(dtype=float)
    n = len(xs)
    if n < 3:
        raise ValueError("Need at least 3 data points for interpolation")
    gx = np.linspace(xs.min(), xs.max(), grid_size)
    gy = np.linspace(ys.min(), ys.max(), grid_size)
    grid = np.empty((grid_size, grid_size))
    for i, gxi in enumerate(gx):
        for j, gyj in enumerate(gy):
            dists = np.sqrt((xs - gxi) ** 2 + (ys - gyj) ** 2)
            exact = np.where(dists < 1e-10)[0]
            if len(exact) > 0:
                grid[i, j] = zs[exact[0]]
            else:
                weights = 1.0 / dists**power
                grid[i, j] = np.sum(weights * zs) / np.sum(weights)
    return DescriptiveResult(
        name="IDW interpolation",
        value=grid.tolist(),
        extra={
            "n_points": n,
            "grid_size": grid_size,
            "power": power,
            "z_min": float(np.nanmin(grid)),
            "z_max": float(np.nanmax(grid)),
            "z_mean": float(np.nanmean(grid)),
            "x_range": [float(xs.min()), float(xs.max())],
            "y_range": [float(ys.min()), float(ys.max())],
        },
    )


aqman = idw_interpolate


def cheatsheet() -> str:
    return "idw_interpolate({}) -> Aquifer/water table interpolation. 'The sea does not like to"
