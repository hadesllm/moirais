"""Spatiotemporal autocorrelation function (space-time Moran's I)."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def st_autocorrelation(
    values: np.ndarray,
    coords: np.ndarray,
    times: np.ndarray,
    time_lags: np.ndarray | None = None,
    bandwidth: float | None = None,
) -> SpatialResult:
    r"""Compute space-time Moran's I across temporal lags.

    For each time lag :math:`\tau`, computes:

    .. math::

        I(\tau) = \frac{n}{\sum_{i}\sum_{j} w_{ij}(\tau)}
        \frac{\sum_{i}\sum_{j} w_{ij}(\tau)(x_i - \bar{x})(x_j - \bar{x})}
        {\sum_{i}(x_i - \bar{x})^2}

    where :math:`w_{ij}(\tau)` is the spatiotemporal weight between
    observations *i* and *j* at temporal lag :math:`\tau`, defined by a
    Gaussian spatial kernel and exact temporal matching.

    Parameters
    ----------
    values : np.ndarray
        Observed values, shape ``(n,)``.
    coords : np.ndarray
        Spatial coordinates, shape ``(n, 2)``.
    times : np.ndarray
        Time index for each observation, shape ``(n,)``.
    time_lags : np.ndarray, optional
        Temporal lags to evaluate. Defaults to unique time differences.
    bandwidth : float, optional
        Spatial bandwidth for Gaussian kernel. Defaults to median
        pairwise distance.

    Returns
    -------
    SpatialResult
        ``statistic`` is the mean space-time Moran's I across lags.
        ``extra["lag_values"]`` contains per-lag I values.

    References
    ----------
    Cliff AD, Ord JK (1981). *Spatial Processes: Models and Applications*.
    Pion, London.

    Rey SJ, Anselin L (2007). PySAL: A Python library of spatial analytical
    methods. *Review of Regional Studies*, 37(1), 5-27.

    Getis A (2005). Spatial autocorrelation. In: Fischer MM, Getis A (eds.)
    *Handbook of Applied Spatial Analysis*. Springer.
    """
    vals = np.asarray(values, dtype=np.float64).ravel()
    xy = np.asarray(coords, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64).ravel()
    n = len(vals)

    if xy.shape != (n, 2):
        raise ValueError("coords must be (n, 2)")
    if t.shape != (n,):
        raise ValueError("times must have same length as values")

    diffs = np.abs(t[:, None] - t[None, :])
    unique_diffs = np.unique(diffs)
    unique_diffs = unique_diffs[unique_diffs > 0]

    if time_lags is not None:
        lags = np.asarray(time_lags, dtype=np.float64)
    else:
        lags = unique_diffs[:10] if len(unique_diffs) > 10 else unique_diffs

    dx = xy[:, 0][:, None] - xy[:, 0][None, :]
    dy = xy[:, 1][:, None] - xy[:, 1][None, :]
    spatial_dist = np.sqrt(dx**2 + dy**2)

    if bandwidth is None:
        upper = spatial_dist[np.triu_indices(n, k=1)]
        bandwidth = float(np.median(upper)) if len(upper) > 0 else 1.0

    z = vals - vals.mean()
    ss = np.sum(z**2)
    if ss == 0:
        return SpatialResult(
            name="ST_Moran_I",
            statistic=0.0,
            p_value=None,
            extra={"lag_values": np.zeros(len(lags)), "lags": lags},
        )

    spatial_w = np.exp(-0.5 * (spatial_dist / bandwidth) ** 2)
    np.fill_diagonal(spatial_w, 0.0)

    lag_I = np.empty(len(lags))
    for k, tau in enumerate(lags):
        temporal_mask = (np.abs(diffs - tau) < 1e-10).astype(np.float64)
        w = spatial_w * temporal_mask
        w_sum = w.sum()
        if w_sum == 0:
            lag_I[k] = 0.0
        else:
            lag_I[k] = (n / w_sum) * np.sum(w * np.outer(z, z)) / ss

    mean_I = float(np.mean(lag_I))

    return SpatialResult(
        name="ST_Moran_I",
        statistic=mean_I,
        p_value=None,
        extra={"lag_values": lag_I, "lags": lags},
    )


def cheatsheet() -> str:
    return "st_autocorrelation({}) -> Spatiotemporal autocorrelation function (space-time Moran's "
