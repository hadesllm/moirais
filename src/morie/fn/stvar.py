"""Spatiotemporal variogram estimation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def st_variogram(
    values: np.ndarray,
    coords: np.ndarray,
    times: np.ndarray,
    n_spatial_bins: int = 15,
    n_temporal_bins: int = 10,
) -> SpatialResult:
    r"""Estimate an empirical spatiotemporal variogram.

    Computes the space-time semivariance:

    .. math::

        \hat{\gamma}(h_s, h_t) = \frac{1}{2|N(h_s, h_t)|}
        \sum_{(i,j) \in N(h_s, h_t)}
        \bigl(Z(s_i, t_i) - Z(s_j, t_j)\bigr)^2

    where :math:`N(h_s, h_t)` is the set of pairs with spatial lag
    :math:`\approx h_s` and temporal lag :math:`\approx h_t`.

    Parameters
    ----------
    values : np.ndarray
        Observed values, shape ``(n,)``.
    coords : np.ndarray
        Spatial coordinates, shape ``(n, 2)``.
    times : np.ndarray
        Time values, shape ``(n,)``.
    n_spatial_bins : int
        Number of spatial distance bins. Default 15.
    n_temporal_bins : int
        Number of temporal lag bins. Default 10.

    Returns
    -------
    SpatialResult
        ``statistic`` is the overall sill (max semivariance).
        ``extra`` contains ``gamma_matrix``, ``spatial_bins``,
        ``temporal_bins``, ``counts``.

    References
    ----------
    Cressie N, Huang H-C (1999). Classes of nonseparable, spatio-temporal
    stationary covariance functions. *Journal of the American Statistical
    Association*, 94(448), 1330-1340.

    Gneiting T (2002). Nonseparable, stationary covariance functions for
    space-time data. *Journal of the American Statistical Association*,
    97(458), 590-600.

    De Cesare L, Myers DE, Posa D (2001). Estimating and modeling
    space-time correlation structures. *Statistics & Probability Letters*,
    51(1), 9-14.
    """
    vals = np.asarray(values, dtype=np.float64).ravel()
    xy = np.asarray(coords, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64).ravel()
    n = len(vals)

    if xy.shape != (n, 2):
        raise ValueError("coords must be (n, 2)")

    dx = xy[:, 0][:, None] - xy[:, 0][None, :]
    dy = xy[:, 1][:, None] - xy[:, 1][None, :]
    spatial_dist = np.sqrt(dx**2 + dy**2)
    temporal_dist = np.abs(t[:, None] - t[None, :])

    upper = np.triu_indices(n, k=1)
    sd_flat = spatial_dist[upper]
    td_flat = temporal_dist[upper]
    sq_diff = 0.5 * (vals[upper[0]] - vals[upper[1]]) ** 2

    s_edges = np.linspace(0, sd_flat.max() * 1.01, n_spatial_bins + 1)
    t_edges = np.linspace(0, td_flat.max() * 1.01, n_temporal_bins + 1)

    gamma = np.zeros((n_spatial_bins, n_temporal_bins))
    counts = np.zeros((n_spatial_bins, n_temporal_bins), dtype=int)

    s_idx = np.digitize(sd_flat, s_edges) - 1
    t_idx = np.digitize(td_flat, t_edges) - 1

    s_idx = np.clip(s_idx, 0, n_spatial_bins - 1)
    t_idx = np.clip(t_idx, 0, n_temporal_bins - 1)

    for k in range(len(sq_diff)):
        si, ti = s_idx[k], t_idx[k]
        gamma[si, ti] += sq_diff[k]
        counts[si, ti] += 1

    with np.errstate(invalid="ignore"):
        gamma = np.where(counts > 0, gamma / counts, np.nan)

    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])

    sill = float(np.nanmax(gamma)) if not np.all(np.isnan(gamma)) else 0.0

    return SpatialResult(
        name="ST_Variogram",
        statistic=sill,
        p_value=None,
        extra={
            "gamma_matrix": gamma,
            "spatial_bins": s_centers,
            "temporal_bins": t_centers,
            "counts": counts,
        },
    )


def cheatsheet() -> str:
    return "st_variogram({}) -> Spatiotemporal variogram estimation."
