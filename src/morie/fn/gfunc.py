# morie.fn — function file (hadesllm/morie)
"""Pair correlation function g(r) for spatial point patterns."""

import numpy as np

from ._containers import DescriptiveResult


def pair_correlation(
    points: np.ndarray, distances: np.ndarray | None = None, n_distances: int = 20, bandwidth: float | None = None
) -> DescriptiveResult:
    """
    Compute the pair correlation function g(r).

    g(r) is the derivative of K(r) normalised by 2*pi*r.
    Under CSR, g(r) = 1. g > 1 indicates clustering at distance r.

    :param points: (n, 2) array of coordinates.
    :param distances: Distances at which to evaluate g (optional).
    :param n_distances: Number of evaluation distances.
    :param bandwidth: Kernel bandwidth for smoothing (default: auto).
    :return: DescriptiveResult with g values and distances.

    References
    ----------
    Stoyan D, Stoyan H (1994). Fractals, Random Shapes and Point Fields.
    John Wiley & Sons.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be (n, 2).")
    n = pts.shape[0]
    x_range = pts[:, 0].max() - pts[:, 0].min()
    y_range = pts[:, 1].max() - pts[:, 1].min()
    area = x_range * y_range if x_range > 0 and y_range > 0 else 1.0
    lam = n / area
    dmat = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dmat, np.inf)
    if distances is None:
        max_d = np.percentile(dmat[dmat < np.inf], 50)
        distances = np.linspace(max_d * 0.05, max_d, n_distances)
    else:
        distances = np.asarray(distances, dtype=np.float64)
    if bandwidth is None:
        bandwidth = float(distances[1] - distances[0]) if len(distances) > 1 else 0.1
    g_vals = np.zeros(len(distances))
    for idx, r in enumerate(distances):
        if r <= 0:
            continue
        kernel_weights = np.exp(-((dmat - r) ** 2) / (2 * bandwidth**2))
        np.fill_diagonal(kernel_weights, 0)
        g_vals[idx] = np.sum(kernel_weights) / (n * (n - 1) * 2 * np.pi * r * bandwidth * np.sqrt(2 * np.pi) * lam)
    if g_vals.max() > 0:
        g_vals = g_vals / g_vals.mean() if g_vals.mean() > 0 else g_vals
    return DescriptiveResult(
        name="pair_correlation",
        value=float(g_vals.mean()) if len(g_vals) > 0 else 0.0,
        extra={"g": g_vals, "distances": distances, "n_points": n, "bandwidth": bandwidth},
    )


gfunc = pair_correlation


def cheatsheet() -> str:
    return "pair_correlation({}) -> Pair correlation function g(r) for spatial point patterns."
