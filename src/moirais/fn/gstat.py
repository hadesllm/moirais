# moirais.fn — function file (hadesllm/moirais)
"""Empirical semivariogram estimation."""

import numpy as np
from scipy.spatial.distance import pdist

from ._containers import DescriptiveResult


def semivariogram(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    n_bins: int = 15,
    max_dist: float | None = None,
) -> DescriptiveResult:
    """
    Compute the empirical (experimental) semivariogram.

    For each distance bin *h*:

    .. math::

        \\hat{\\gamma}(h) = \\frac{1}{2|N(h)|}
                            \\sum_{(i,j) \\in N(h)} (x_i - x_j)^2

    where :math:`N(h)` is the set of observation pairs with inter-point
    distance falling in the bin centred on *h*.

    :param coords: Observation coordinates, shape (n, 2).
    :param values: Observed values, shape (n,).
    :param n_bins: Number of distance bins. Default 15.
    :param max_dist: Maximum distance to consider. Default half the maximum
        inter-point distance.
    :return: :class:`DescriptiveResult` with extra keys ``lags``,
        ``semivariance``, and ``counts`` (number of pairs per bin).
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Matheron, G. (1963). Principles of geostatistics. *Economic Geology*,
    58(8), 1246-1266. https://doi.org/10.2113/gsecongeo.58.8.1246

    Cressie, N. A. C. (1993). *Statistics for Spatial Data* (rev. ed.).
    Wiley.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    n = int(values) if values.ndim == 0 else len(values)
    if coords.shape[0] != n:
        raise ValueError(f"coords rows ({coords.shape[0]}) != values length ({n}).")
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coords must have shape (n, 2) or (n, d).")

    dists = pdist(coords)
    val_diffs_sq = pdist(values[:, None]) ** 2

    if max_dist is None:
        max_dist = float(dists.max()) / 2.0

    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    lags = np.zeros(n_bins)
    semivariance = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (dists > lo) & (dists <= hi) if b > 0 else (dists >= lo) & (dists <= hi)
        cnt = int(mask.sum())
        counts[b] = cnt
        if cnt > 0:
            lags[b] = float(dists[mask].mean())
            semivariance[b] = float(val_diffs_sq[mask].mean()) / 2.0
        else:
            lags[b] = (lo + hi) / 2.0

    return DescriptiveResult(
        name="Semivariogram",
        value=None,
        extra={
            "lags": lags,
            "semivariance": semivariance,
            "counts": counts,
            "n_bins": n_bins,
            "max_dist": max_dist,
            "n": n,
        },
    )


gstat_fn = semivariogram


def cheatsheet() -> str:
    return "semivariogram({}) -> Empirical semivariogram estimation."
