# morie.fn -- function file (hadesllm/morie)
"""Spatial point process intensity estimation (kernel)."""

import numpy as np

from ._containers import DescriptiveResult


def point_process_intensity(points: np.ndarray, bandwidth: float = 1.0, grid_size: int = 50) -> DescriptiveResult:
    """
    Kernel density estimation for spatial point process intensity.

    Uses a Gaussian kernel to estimate the spatial intensity function
    on a regular grid.

    :param points: (n, 2) array of (x, y) coordinates.
    :param bandwidth: Kernel bandwidth (default 1.0).
    :param grid_size: Number of grid cells per dimension.
    :return: DescriptiveResult with intensity grid in extra.
    :raises ValueError: If points not 2D or bandwidth <= 0.

    References
    ----------
    Diggle PJ (2003). Statistical Analysis of Spatial Point Patterns.
    2nd ed. Arnold.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be (n, 2).")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be > 0.")
    n = pts.shape[0]
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    pad = bandwidth
    gx = np.linspace(x_min - pad, x_max + pad, grid_size)
    gy = np.linspace(y_min - pad, y_max + pad, grid_size)
    GX, GY = np.meshgrid(gx, gy)
    intensity = np.zeros_like(GX)
    for i in range(n):
        dx = GX - pts[i, 0]
        dy = GY - pts[i, 1]
        intensity += np.exp(-(dx**2 + dy**2) / (2 * bandwidth**2))
    intensity /= 2 * np.pi * bandwidth**2 * n
    area = (x_max - x_min + 2 * pad) * (y_max - y_min + 2 * pad)
    return DescriptiveResult(
        name="point_process_intensity",
        value=float(n / area),
        extra={
            "intensity": intensity,
            "grid_x": gx,
            "grid_y": gy,
            "n_points": n,
            "bandwidth": bandwidth,
            "mean_intensity": float(n / area),
        },
    )


pproc = point_process_intensity


def cheatsheet() -> str:
    return "point_process_intensity({}) -> Spatial point process intensity estimation (kernel)."
