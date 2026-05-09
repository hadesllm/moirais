"""Empirical semivariogram estimation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def empirical_semivariogram(Z, coords, n_lags=15, max_lag=None):
    """Compute the empirical semivariogram gamma(h).

    .. epigraph:: "Rise, Tarnished." -- Two Fingers, Elden Ring

    Parameters
    ----------
    Z : array_like
        Observed values.
    coords : array_like
        Coordinates, shape ``(n, 2)``.
    n_lags : int
        Number of lag bins.
    max_lag : float, optional
        Maximum lag distance (default: half max distance).

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
    gamma = np.zeros(n_lags)
    counts = np.zeros(n_lags, dtype=int)

    for k in range(n_lags):
        mask = (edges[k] < D) & (edges[k + 1] >= D)
        mask_upper = np.triu(mask, k=1)
        pairs = np.argwhere(mask_upper)
        if len(pairs) > 0:
            diffs = Z[pairs[:, 0]] - Z[pairs[:, 1]]
            gamma[k] = 0.5 * np.mean(diffs**2)
            counts[k] = len(pairs)

    return DescriptiveResult(
        name="empirical_semivariogram",
        value=float(gamma.max()) if gamma.max() > 0 else 0.0,
        extra={
            "lag_distances": mids.tolist(),
            "gamma_values": gamma.tolist(),
            "pair_counts": counts.tolist(),
            "n_lags": n_lags,
        },
    )


sgemp = empirical_semivariogram


def cheatsheet() -> str:
    return "empirical_semivariogram({}) -> Empirical semivariogram estimation."
