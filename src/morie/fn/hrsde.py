# morie.fn — function file (hadesllm/morie)
"""Home range estimation via kernel density (Worton 1989)."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def home_range_kde(
    coords: np.ndarray,
    bandwidth: float | None = None,
    level: float = 0.95,
    n_grid: int = 100,
) -> SpatialResult:
    r"""Estimate animal home range using fixed kernel density.

    Estimates the utilization distribution (UD) and returns the area
    of the ``level`` isopleth contour:

    .. math::

        \hat{f}(x, y) = \frac{1}{n h^2}
        \sum_{i=1}^n K\!\left(\frac{\|(x,y) - (x_i, y_i)\|}{h}\right)

    The home range area is the region containing ``level`` proportion
    of the UD volume, computed by sorting grid cells by density and
    accumulating until the threshold is reached.

    Parameters
    ----------
    coords : np.ndarray
        Location fixes, shape ``(n, 2)``.
    bandwidth : float, optional
        Fixed bandwidth *h*. Default: reference bandwidth
        (Silverman rule).
    level : float
        Isopleth level (e.g. 0.95 for 95% home range). Default 0.95.
    n_grid : int
        Grid resolution per axis. Default 100.

    Returns
    -------
    SpatialResult
        ``statistic`` is the estimated home range area (in coordinate
        units squared).
        ``extra`` contains ``density`` (2-D), ``x_grid``, ``y_grid``,
        ``bandwidth``, ``isopleth_mask``.

    References
    ----------
    Worton BJ (1989). Kernel methods for estimating the utilization
    distribution in home-range studies. *Ecology*, 70(1), 164-168.

    Seaman DE, Powell RA (1996). An evaluation of the accuracy of
    kernel density estimators for home range analysis. *Ecology*,
    77(7), 2075-2085.

    Kie JG, Matthiopoulos J, Fieberg J, Powell RA, Cagnacci F,
    Mitchell MS, Gaillard J-M, Moorcroft PR (2010). The home-range
    concept: are traditional estimators still relevant with modern
    telemetry technology? *Philosophical Transactions of the Royal
    Society B*, 365(1550), 2221-2231.
    """
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("coords must be (n, 2)")

    n = pts.shape[0]

    if bandwidth is None:
        sx = np.std(pts[:, 0], ddof=1)
        sy = np.std(pts[:, 1], ddof=1)
        bandwidth = ((sx + sy) / 2.0) * n ** (-1.0 / 6.0)
    bandwidth = max(bandwidth, 1e-10)

    margin = 4 * bandwidth
    x_grid = np.linspace(pts[:, 0].min() - margin, pts[:, 0].max() + margin, n_grid)
    y_grid = np.linspace(pts[:, 1].min() - margin, pts[:, 1].max() + margin, n_grid)
    cell_area = (x_grid[1] - x_grid[0]) * (y_grid[1] - y_grid[0])

    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
    density = np.zeros_like(xx)

    for i in range(n):
        dx = (xx - pts[i, 0]) / bandwidth
        dy = (yy - pts[i, 1]) / bandwidth
        density += np.exp(-0.5 * (dx**2 + dy**2))

    density /= n * 2 * np.pi * bandwidth**2

    flat = density.ravel()
    sort_idx = np.argsort(flat)[::-1]
    cumsum = np.cumsum(flat[sort_idx] * cell_area)
    total = cumsum[-1] if len(cumsum) > 0 else 1.0
    cutoff_idx = np.searchsorted(cumsum, level * total)
    n_cells = cutoff_idx + 1
    area = float(n_cells * cell_area)

    mask = np.zeros(len(flat), dtype=bool)
    mask[sort_idx[:n_cells]] = True
    isopleth_mask = mask.reshape(density.shape)

    return SpatialResult(
        name="Home_Range_KDE",
        statistic=area,
        p_value=None,
        extra={
            "density": density,
            "x_grid": x_grid,
            "y_grid": y_grid,
            "bandwidth": bandwidth,
            "level": level,
            "isopleth_mask": isopleth_mask,
        },
    )


def cheatsheet() -> str:
    return "home_range_kde({}) -> Home range estimation via kernel density (Worton 1989)."
