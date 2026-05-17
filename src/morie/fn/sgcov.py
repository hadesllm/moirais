"""Spatial covariance function estimation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def covariance_function_estimate(Z, coords, max_lag=None, n_lags=15):
    """Estimate the spatial covariance function C(h).

    .. epigraph:: Number rules the universe. -- Pythagoras

    Parameters
    ----------
    Z : array_like
        Observed values.
    coords : array_like
        Coordinates, shape ``(n, 2)``.
    max_lag : float, optional
        Maximum lag distance. Defaults to half the max pairwise distance.
    n_lags : int
        Number of lag bins.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)

    D = squareform(pdist(coords))
    if max_lag is None:
        max_lag = D.max() / 2.0

    edges = np.linspace(0, max_lag, n_lags + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    z_centered = Z - Z.mean()

    cov_values = np.zeros(n_lags)
    counts = np.zeros(n_lags, dtype=int)
    for k in range(n_lags):
        mask = (edges[k] < D) & (edges[k + 1] >= D)
        mask_upper = np.triu(mask, k=1)
        pairs = np.argwhere(mask_upper)
        if len(pairs) > 0:
            cov_values[k] = np.mean(z_centered[pairs[:, 0]] * z_centered[pairs[:, 1]])
            counts[k] = len(pairs)

    c0 = float(np.var(Z))

    return DescriptiveResult(
        name="covariance_function_estimate",
        value=c0,
        extra={
            "lag_distances": mids.tolist(),
            "covariance_values": cov_values.tolist(),
            "pair_counts": counts.tolist(),
            "variance_c0": c0,
        },
    )


sgcov = covariance_function_estimate


def cheatsheet() -> str:
    return "covariance_function_estimate({}) -> Spatial covariance function estimation."
