"""Strict stationarity test via subregion comparison."""

from __future__ import annotations

from ._containers import DescriptiveResult


def strict_stationarity_test(Z, coords, n_sub=4):
    """Test strict stationarity by comparing distributions across subregions.

    Divides the study area into *n_sub* x *n_sub* subregions and runs
    Kolmogorov-Smirnov tests between all pairs.

    .. epigraph:: "You shall not pass!" -- Gandalf, Lord of the Rings

    Parameters
    ----------
    Z : array_like
        Observed values, shape ``(n,)``.
    coords : array_like
        Coordinates, shape ``(n, 2)``.
    n_sub : int
        Grid divisions per axis (default 4).

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np
    from scipy import stats

    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    n = len(Z)

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    xedges = np.linspace(xmin, ymax, n_sub + 1)
    yedges = np.linspace(ymin, ymax, n_sub + 1)

    xedges = np.linspace(xmin, xmax, n_sub + 1)
    yedges = np.linspace(ymin, ymax, n_sub + 1)

    groups = []
    for i in range(n_sub):
        for j in range(n_sub):
            mask = (
                (coords[:, 0] >= xedges[i])
                & (coords[:, 0] < xedges[i + 1])
                & (coords[:, 1] >= yedges[j])
                & (coords[:, 1] < yedges[j + 1])
            )
            vals = Z[mask]
            if len(vals) > 1:
                groups.append(vals)

    ks_stats = []
    p_values = []
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            s, p = stats.ks_2samp(groups[a], groups[b])
            ks_stats.append(s)
            p_values.append(p)

    mean_ks = float(np.mean(ks_stats)) if ks_stats else 0.0
    min_p = float(np.min(p_values)) if p_values else 1.0

    return DescriptiveResult(
        name="strict_stationarity_test",
        value=mean_ks,
        extra={
            "mean_ks": mean_ks,
            "min_p_value": min_p,
            "n_comparisons": len(ks_stats),
            "n_subregions": len(groups),
            "stationary": min_p > 0.05,
        },
    )


sgstr = strict_stationarity_test


def cheatsheet() -> str:
    return "strict_stationarity_test({}) -> Strict stationarity test via subregion comparison."
