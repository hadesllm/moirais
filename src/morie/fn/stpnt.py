"""Space-time point pattern intensity estimation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def spacetime_intensity(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    n_xbins: int = 10,
    n_ybins: int = 10,
    n_tbins: int = 10,
    bandwidth_s: float | None = None,
    bandwidth_t: float | None = None,
) -> SpatialResult:
    r"""
    Estimate space-time point pattern intensity via kernel smoothing.

    Uses a separable Gaussian kernel in space and time to produce a 3-D
    intensity estimate :math:`\hat{\lambda}(s, t)` over a regular grid.

    .. math::

        \hat{\lambda}(s, t) = \sum_{i=1}^{n} K_s\!\bigl(\|s - s_i\|\bigr)
                              \, K_t\!\bigl(|t - t_i|\bigr)

    where :math:`K_s` and :math:`K_t` are Gaussian kernels with bandwidths
    ``bandwidth_s`` and ``bandwidth_t``.

    Parameters
    ----------
    x, y : np.ndarray
        (n,) spatial coordinates of events.
    t : np.ndarray
        (n,) temporal coordinates.
    n_xbins, n_ybins, n_tbins : int
        Grid resolution in each dimension.
    bandwidth_s : float or None
        Spatial bandwidth. If None, uses Silverman's rule on pooled coords.
    bandwidth_t : float or None
        Temporal bandwidth. If None, uses Silverman's rule on time coords.

    Returns
    -------
    SpatialResult
        statistic = peak intensity, extra has ``grid`` (3-D array),
        ``x_edges``, ``y_edges``, ``t_edges``.

    References
    ----------
    Diggle PJ (2013). *Statistical Analysis of Spatial and Spatio-Temporal
    Point Patterns*. 3rd ed. CRC Press. Chapters 6--7.

    Gabriel E, Diggle PJ (2009). Second-order analysis of inhomogeneous
    spatio-temporal point process data. *Statistica Neerlandica*, 63(1),
    43--51. doi:10.1111/j.1467-9574.2008.00407.x

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> res = spacetime_intensity(rng.uniform(0, 1, 100),
    ...     rng.uniform(0, 1, 100), rng.uniform(0, 10, 100))
    >>> res.statistic > 0
    True
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    n = len(x)
    if len(y) != n or len(t) != n:
        raise ValueError("x, y, t must have the same length.")

    def _silverman(v: np.ndarray) -> float:
        s = float(np.std(v, ddof=1))
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        h = 0.9 * min(s, iqr / 1.34) * n ** (-0.2)
        return max(h, 1e-10)

    if bandwidth_s is None:
        bandwidth_s = _silverman(np.concatenate([x, y]))
    if bandwidth_t is None:
        bandwidth_t = _silverman(t)

    x_edges = np.linspace(float(x.min()), float(x.max()), n_xbins + 1)
    y_edges = np.linspace(float(y.min()), float(y.max()), n_ybins + 1)
    t_edges = np.linspace(float(t.min()), float(t.max()), n_tbins + 1)
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    tc = 0.5 * (t_edges[:-1] + t_edges[1:])

    grid = np.zeros((n_xbins, n_ybins, n_tbins))
    for i in range(n):
        wx = np.exp(-0.5 * ((xc - x[i]) / bandwidth_s) ** 2)
        wy = np.exp(-0.5 * ((yc - y[i]) / bandwidth_s) ** 2)
        wt = np.exp(-0.5 * ((tc - t[i]) / bandwidth_t) ** 2)
        grid += wx[:, None, None] * wy[None, :, None] * wt[None, None, :]

    norm = (2 * np.pi) ** 1.5 * bandwidth_s**2 * bandwidth_t
    grid /= norm

    return SpatialResult(
        name="spacetime_intensity",
        statistic=float(grid.max()),
        extra={
            "grid": grid,
            "x_edges": x_edges,
            "y_edges": y_edges,
            "t_edges": t_edges,
            "bandwidth_s": bandwidth_s,
            "bandwidth_t": bandwidth_t,
            "n": n,
        },
    )


stpnt = spacetime_intensity


def cheatsheet() -> str:
    return "spacetime_intensity({}) -> Space-time point pattern intensity estimation."
