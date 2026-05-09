"""Weak (second-order) stationarity test via variogram."""

from __future__ import annotations

from ._containers import DescriptiveResult


def weak_stationarity_test(Z, coords, n_lags=10):
    """Test weak stationarity by checking variogram stability across subregions.

    Computes empirical variograms in spatial quadrants and compares them.

    .. epigraph:: "Always." -- Severus Snape, Harry Potter

    Parameters
    ----------
    Z : array_like
        Observed values, shape ``(n,)``.
    coords : array_like
        Coordinates, shape ``(n, 2)``.
    n_lags : int
        Number of lag bins.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)

    cx, cy = np.median(coords[:, 0]), np.median(coords[:, 1])
    quadrants = [
        (coords[:, 0] <= cx) & (coords[:, 1] <= cy),
        (coords[:, 0] > cx) & (coords[:, 1] <= cy),
        (coords[:, 0] <= cx) & (coords[:, 1] > cy),
        (coords[:, 0] > cx) & (coords[:, 1] > cy),
    ]

    def _emp_vario(z, xy, n_l):
        from scipy.spatial.distance import pdist, squareform

        D = squareform(pdist(xy))
        max_d = D.max() / 2
        edges = np.linspace(0, max_d, n_l + 1)
        gamma = np.zeros(n_l)
        for k in range(n_l):
            mask = (edges[k] < D) & (edges[k + 1] >= D)
            pairs = np.argwhere(np.triu(mask, k=1))
            if len(pairs) > 0:
                diffs = z[pairs[:, 0]] - z[pairs[:, 1]]
                gamma[k] = 0.5 * np.mean(diffs**2)
        return gamma, 0.5 * (edges[:-1] + edges[1:])

    variograms = []
    for q in quadrants:
        if q.sum() > 5:
            g, _ = _emp_vario(Z[q], coords[q], n_lags)
            variograms.append(g)

    if len(variograms) < 2:
        return DescriptiveResult(
            name="weak_stationarity_test",
            value=0.0,
            extra={"stationary": True, "reason": "insufficient_subregions"},
        )

    max_diff = 0.0
    for i in range(len(variograms)):
        for j in range(i + 1, len(variograms)):
            diff = np.max(np.abs(variograms[i] - variograms[j]))
            max_diff = max(max_diff, diff)

    var_z = float(np.var(Z))
    relative = max_diff / var_z if var_z > 0 else 0.0

    return DescriptiveResult(
        name="weak_stationarity_test",
        value=relative,
        extra={
            "max_variogram_diff": float(max_diff),
            "relative_diff": relative,
            "n_quadrants": len(variograms),
            "stationary": relative < 0.5,
        },
    )


sgwks = weak_stationarity_test


def cheatsheet() -> str:
    return "weak_stationarity_test({}) -> Weak (second-order) stationarity test via variogram."
