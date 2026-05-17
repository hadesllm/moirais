"""Pair correlation function g(r)."""

from __future__ import annotations

from ._containers import DescriptiveResult


def pair_correlation_function(points, window, r_values=None, bandwidth=None):
    """Estimate the pair correlation function g(r).

    g(r) = 1 under CSR, > 1 clustering, < 1 inhibition at distance r.

    .. epigraph:: Statistics is the grammar of science. -- Karl Pearson

    Parameters
    ----------
    points : array_like
        Point coordinates, shape ``(n, 2)``.
    window : tuple
        ``(xmin, xmax, ymin, ymax)``.
    r_values : array_like, optional
        Distances at which to evaluate g(r).
    bandwidth : float, optional
        Kernel bandwidth for smoothing.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np
    from scipy.spatial.distance import pdist

    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    xmin, xmax, ymin, ymax = window
    area = (xmax - xmin) * (ymax - ymin)
    lam = n / area

    dists = pdist(pts)

    if r_values is None:
        max_r = min(xmax - xmin, ymax - ymin) / 4
        r_values = np.linspace(max_r * 0.05, max_r, 25)
    else:
        r_values = np.asarray(r_values, dtype=np.float64)

    if bandwidth is None:
        bandwidth = (r_values[1] - r_values[0]) if len(r_values) > 1 else 0.1

    g = np.zeros(len(r_values))
    for ri, r in enumerate(r_values):
        kernel_vals = np.exp(-((dists - r) ** 2) / (2 * bandwidth**2))
        kernel_vals /= bandwidth * np.sqrt(2 * np.pi)
        ring_area = 2 * np.pi * r if r > 0 else 1.0
        expected = 0.5 * n * (n - 1) * lam * ring_area * (1.0 / area) if area > 0 else 1.0
        g[ri] = kernel_vals.sum() / expected if expected > 0 else 0.0

    return DescriptiveResult(
        name="pair_correlation_function",
        value=float(np.max(g)) if len(g) > 0 else 0.0,
        extra={
            "r_values": r_values.tolist(),
            "g_values": g.tolist(),
            "intensity": lam,
        },
    )


sgpcf = pair_correlation_function


def cheatsheet() -> str:
    return "pair_correlation_function({}) -> Pair correlation function g(r)."
