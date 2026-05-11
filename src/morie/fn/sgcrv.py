"""Cross-variogram estimation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def cross_variogram(Z1, Z2, coords, n_lags=15, max_lag=None):
    """Compute the cross-semivariogram gamma_12(h) for two variables.

    .. epigraph:: "I am the storm that is approaching." -- Vergil, Devil May Cry

    Parameters
    ----------
    Z1 : array_like
        First variable values.
    Z2 : array_like
        Second variable values.
    coords : array_like
        Shared coordinates, shape ``(n, 2)``.
    n_lags : int
        Number of lag bins.
    max_lag : float, optional
        Maximum lag distance.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    Z1 = np.asarray(Z1, dtype=np.float64).ravel()
    Z2 = np.asarray(Z2, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)

    D = squareform(pdist(coords))
    if max_lag is None:
        max_lag = D.max() / 2.0

    edges = np.linspace(0, max_lag, n_lags + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    gamma12 = np.zeros(n_lags)
    counts = np.zeros(n_lags, dtype=int)

    for k in range(n_lags):
        mask = np.triu((edges[k] < D) & (edges[k + 1] >= D), k=1)
        pairs = np.argwhere(mask)
        if len(pairs) > 0:
            d1 = Z1[pairs[:, 0]] - Z1[pairs[:, 1]]
            d2 = Z2[pairs[:, 0]] - Z2[pairs[:, 1]]
            gamma12[k] = 0.5 * np.mean(d1 * d2)
            counts[k] = len(pairs)

    return DescriptiveResult(
        name="cross_variogram",
        value=float(np.max(np.abs(gamma12))) if len(gamma12) > 0 else 0.0,
        extra={
            "lag_distances": mids.tolist(),
            "gamma12_values": gamma12.tolist(),
            "pair_counts": counts.tolist(),
            "n_lags": n_lags,
        },
    )


sgcrv = cross_variogram


def cheatsheet() -> str:
    return "cross_variogram({}) -> Cross-variogram estimation."
