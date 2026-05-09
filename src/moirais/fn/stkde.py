"""Spatiotemporal kernel density estimation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def st_kde(
    coords: np.ndarray,
    times: np.ndarray,
    x_grid: np.ndarray | None = None,
    y_grid: np.ndarray | None = None,
    t_grid: np.ndarray | None = None,
    bw_spatial: float | None = None,
    bw_temporal: float | None = None,
    n_grid: int = 25,
) -> SpatialResult:
    r"""Spatiotemporal kernel density estimation with Gaussian kernels.

    Estimates density at grid points ``(x, y, t)`` via:

    .. math::

        \hat{f}(x, y, t) = \frac{1}{n \, h_s^2 \, h_t}
        \sum_{i=1}^{n}
        K_s\!\left(\frac{\|(x,y) - (x_i, y_i)\|}{h_s}\right)
        K_t\!\left(\frac{t - t_i}{h_t}\right)

    where :math:`K_s` and :math:`K_t` are Gaussian kernels with
    bandwidths :math:`h_s` (spatial) and :math:`h_t` (temporal).

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates, shape ``(n, 2)``.
    times : np.ndarray
        Time values, shape ``(n,)``.
    x_grid, y_grid, t_grid : np.ndarray, optional
        1-D grid vectors. Defaults to ``n_grid`` equally spaced points.
    bw_spatial : float, optional
        Spatial bandwidth. Default: Silverman rule on pooled coords.
    bw_temporal : float, optional
        Temporal bandwidth. Default: Silverman rule on time.
    n_grid : int
        Number of grid points per dimension. Default 25.

    Returns
    -------
    SpatialResult
        ``statistic`` is the peak density value.
        ``extra`` contains ``density`` (3-D array), ``x_grid``,
        ``y_grid``, ``t_grid``.

    References
    ----------
    Silverman BW (1986). *Density Estimation for Statistics and Data
    Analysis*. Chapman & Hall.

    Nakaya T, Yano K (2010). Visualising crime clusters in a space-time
    cube: An exploratory data-analysis approach using space-time kernel
    density estimation and scan statistics. *Transactions in GIS*,
    14(3), 223-239.

    Brunsdon C, Corcoran J, Higgs G (2007). Visualising space and time
    in crime patterns. *Applied Geography*, 27(3), 148-164.
    """
    xy = np.asarray(coords, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64).ravel()
    n = len(t)

    if xy.shape != (n, 2):
        raise ValueError("coords must be (n, 2)")

    def _silverman(data):
        s = np.std(data, ddof=1)
        iqr = np.subtract(*np.percentile(data, [75, 25]))
        h = 0.9 * min(s, iqr / 1.34) * n ** (-0.2) if iqr > 0 else 0.9 * s * n ** (-0.2)
        return max(h, 1e-10)

    if bw_spatial is None:
        bw_spatial = (_silverman(xy[:, 0]) + _silverman(xy[:, 1])) / 2.0
    if bw_temporal is None:
        bw_temporal = _silverman(t)

    if x_grid is None:
        margin = 3 * bw_spatial
        x_grid = np.linspace(xy[:, 0].min() - margin, xy[:, 0].max() + margin, n_grid)
    if y_grid is None:
        margin = 3 * bw_spatial
        y_grid = np.linspace(xy[:, 1].min() - margin, xy[:, 1].max() + margin, n_grid)
    if t_grid is None:
        margin = 3 * bw_temporal
        t_grid = np.linspace(t.min() - margin, t.max() + margin, n_grid)

    density = np.zeros((len(x_grid), len(y_grid), len(t_grid)))
    norm = n * bw_spatial**2 * bw_temporal * (2 * np.pi) ** 1.5

    for idx in range(n):
        dx = (x_grid - xy[idx, 0]) / bw_spatial
        dy = (y_grid - xy[idx, 1]) / bw_spatial
        dt = (t_grid - t[idx]) / bw_temporal
        kx = np.exp(-0.5 * dx**2)
        ky = np.exp(-0.5 * dy**2)
        kt = np.exp(-0.5 * dt**2)
        density += kx[:, None, None] * ky[None, :, None] * kt[None, None, :]

    density /= norm

    return SpatialResult(
        name="ST_KDE",
        statistic=float(density.max()),
        p_value=None,
        extra={
            "density": density,
            "x_grid": np.asarray(x_grid),
            "y_grid": np.asarray(y_grid),
            "t_grid": np.asarray(t_grid),
            "bw_spatial": bw_spatial,
            "bw_temporal": bw_temporal,
        },
    )


def cheatsheet() -> str:
    return "st_kde({}) -> Spatiotemporal kernel density estimation."
