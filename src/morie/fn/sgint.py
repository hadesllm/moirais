"""Intensity estimation for point patterns."""

from __future__ import annotations

from ._containers import DescriptiveResult


def intensity_estimate(points, window, method="kernel", bandwidth=None, grid_n=50):
    """Estimate point pattern intensity lambda(s).

    .. epigraph:: We must know. We will know. -- David Hilbert

    Parameters
    ----------
    points : array_like
        Point coordinates, shape ``(n, 2)``.
    window : tuple
        ``(xmin, xmax, ymin, ymax)``.
    method : str
        ``'kernel'`` for KDE or ``'quadrat'`` for quadrat counting.
    bandwidth : float, optional
        Kernel bandwidth. Auto if None.
    grid_n : int
        Grid resolution.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    xmin, xmax, ymin, ymax = window
    area = (xmax - xmin) * (ymax - ymin)

    if method == "kernel":
        if bandwidth is None:
            bandwidth = 0.15 * np.sqrt(area / n) if n > 0 else 1.0
        gx = np.linspace(xmin, xmax, grid_n)
        gy = np.linspace(ymin, ymax, grid_n)
        gxx, gyy = np.meshgrid(gx, gy)
        intensity = np.zeros((grid_n, grid_n))
        for k in range(n):
            d2 = (gxx - pts[k, 0]) ** 2 + (gyy - pts[k, 1]) ** 2
            intensity += np.exp(-d2 / (2 * bandwidth**2))
        intensity /= 2 * np.pi * bandwidth**2
        global_intensity = n / area
    else:
        nx = ny = int(np.sqrt(grid_n))
        xedges = np.linspace(xmin, xmax, nx + 1)
        yedges = np.linspace(ymin, ymax, ny + 1)
        cell_area = ((xmax - xmin) / nx) * ((ymax - ymin) / ny)
        intensity = np.zeros((ny, nx))
        for k in range(n):
            ix = min(int((pts[k, 0] - xmin) / (xmax - xmin) * nx), nx - 1)
            iy = min(int((pts[k, 1] - ymin) / (ymax - ymin) * ny), ny - 1)
            intensity[iy, ix] += 1
        intensity /= cell_area
        global_intensity = n / area

    return DescriptiveResult(
        name="intensity_estimate",
        value=float(global_intensity),
        extra={
            "intensity_grid": intensity,
            "global_intensity": float(global_intensity),
            "method": method,
            "n_points": n,
        },
    )


sgint = intensity_estimate


def cheatsheet() -> str:
    return "intensity_estimate({}) -> Intensity estimation for point patterns."
