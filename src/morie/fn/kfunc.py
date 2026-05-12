# morie.fn — function file (hadesllm/morie)
"""Ripley's K function for spatial point patterns."""

import numpy as np

from ._containers import DescriptiveResult


def ripley_k(points: np.ndarray, distances: np.ndarray | None = None, n_distances: int = 20) -> DescriptiveResult:
    r"""
    Compute Ripley's K function for a spatial point pattern.

    .. math::

        \\hat{K}(d) = \\frac{|A|}{n^2} \\sum_{i \\neq j}
        \\mathbf{1}(\\|s_i - s_j\\| \\le d)

    :param points: (n, 2) array of coordinates.
    :param distances: Distances at which to evaluate K (optional).
    :param n_distances: Number of evenly spaced distances if not provided.
    :return: DescriptiveResult with K values and distances.

    References
    ----------
    Ripley BD (1976). The second-order analysis of stationary point
    processes. Journal of Applied Probability, 13(2), 255-266.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be (n, 2).")
    n = pts.shape[0]
    x_range = pts[:, 0].max() - pts[:, 0].min()
    y_range = pts[:, 1].max() - pts[:, 1].min()
    area = x_range * y_range if x_range > 0 and y_range > 0 else 1.0
    dists = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    if distances is None:
        max_d = dists.max() / 2
        distances = np.linspace(0, max_d, n_distances)
    else:
        distances = np.asarray(distances, dtype=np.float64)
    K_vals = np.zeros(len(distances))
    for idx, d in enumerate(distances):
        count = np.sum(dists <= d) - n
        K_vals[idx] = area / (n * n) * count
    csr_K = np.pi * distances**2
    return DescriptiveResult(
        name="ripley_k",
        value=float(K_vals[-1]) if len(K_vals) > 0 else 0.0,
        extra={"K": K_vals, "distances": distances, "K_csr": csr_K, "n_points": n, "area": area},
    )


kfunc = ripley_k


def cheatsheet() -> str:
    return "ripley_k({}) -> Ripley's K function for spatial point patterns."
