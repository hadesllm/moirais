"""Second-order intensity estimation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def second_order_intensity(points, window, bandwidth=None):
    """Estimate the second-order intensity lambda_2(s1, s2).

    Returns a kernel-smoothed estimate of the product density.

    .. epigraph:: You have power over your mind, not outside events. -- Marcus Aurelius

    Parameters
    ----------
    points : array_like
        Point coordinates, shape ``(n, 2)``.
    window : tuple
        ``(xmin, xmax, ymin, ymax)``.
    bandwidth : float, optional
        Kernel bandwidth.

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
    if bandwidth is None:
        bandwidth = np.std(dists) * 0.2 if len(dists) > 0 else 1.0

    r_vals = np.linspace(0.01, dists.max() / 2 if len(dists) > 0 else 1.0, 30)
    lambda2 = np.zeros(len(r_vals))

    for ri, r in enumerate(r_vals):
        kernel = np.exp(-((dists - r) ** 2) / (2 * bandwidth**2))
        kernel /= bandwidth * np.sqrt(2 * np.pi)
        ring_area = 2 * np.pi * r
        expected_pairs = 0.5 * n * (n - 1) / area
        lambda2[ri] = kernel.sum() / (ring_area * expected_pairs) * lam**2 if ring_area * expected_pairs > 0 else 0.0

    return DescriptiveResult(
        name="second_order_intensity",
        value=float(lam**2),
        extra={
            "r_values": r_vals.tolist(),
            "lambda2_values": lambda2.tolist(),
            "first_order_intensity": lam,
        },
    )


sg2nd = second_order_intensity


def cheatsheet() -> str:
    return "second_order_intensity({}) -> Second-order intensity estimation."
