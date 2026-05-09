"""Spatial autocorrelation function per lag."""

from __future__ import annotations

from ._containers import DescriptiveResult


def spatial_acf(Z, coords, lags=None, n_lags=10):
    """Compute spatial autocorrelation at specified lag distances.

    Returns Moran-style autocorrelation per distance band.

    .. epigraph:: "I am thou, thou art I." -- Persona

    Parameters
    ----------
    Z : array_like
        Observed values.
    coords : array_like
        Coordinates, shape ``(n, 2)``.
    lags : array_like, optional
        Lag bin edges. Auto-computed if None.
    n_lags : int
        Number of lag bins if lags is None.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    D = squareform(pdist(coords))

    if lags is None:
        max_d = D.max() / 2.0
        edges = np.linspace(0, max_d, n_lags + 1)
    else:
        edges = np.asarray(lags, dtype=np.float64)

    z = Z - Z.mean()
    ss = np.sum(z**2)
    n = len(Z)

    acf_vals = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1, dtype=int)

    for k in range(len(edges) - 1):
        mask = (edges[k] < D) & (edges[k + 1] >= D)
        np.fill_diagonal(mask, False)
        w_sum = mask.sum()
        if w_sum > 0 and ss > 0:
            acf_vals[k] = (n / w_sum) * np.sum(mask * np.outer(z, z)) / ss
            counts[k] = w_sum // 2

    mids = 0.5 * (edges[:-1] + edges[1:])

    return DescriptiveResult(
        name="spatial_acf",
        value=float(acf_vals[0]) if len(acf_vals) > 0 else 0.0,
        extra={
            "lag_distances": mids.tolist(),
            "acf_values": acf_vals.tolist(),
            "pair_counts": counts.tolist(),
        },
    )


sgacf = spatial_acf


def cheatsheet() -> str:
    return "spatial_acf({}) -> Spatial autocorrelation function per lag."
