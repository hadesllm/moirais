"""Edge correction weights for point patterns."""

from __future__ import annotations

from ._containers import DescriptiveResult


def edge_correction(points, window, method="ripley"):
    """Compute edge correction weights for point pattern analysis.

    .. epigraph:: To understand God's thoughts we must study statistics. -- Florence Nightingale

    Parameters
    ----------
    points : array_like
        Point coordinates, shape ``(n, 2)``.
    window : tuple
        ``(xmin, xmax, ymin, ymax)``.
    method : str
        ``'ripley'`` (isotropic) or ``'border'``.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    xmin, xmax, ymin, ymax = window

    border_dists = np.minimum(
        np.minimum(pts[:, 0] - xmin, xmax - pts[:, 0]),
        np.minimum(pts[:, 1] - ymin, ymax - pts[:, 1]),
    )

    if method == "ripley":
        from scipy.spatial.distance import pdist, squareform

        D = squareform(pdist(pts))
        weights = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = D[i, j]
                    bd = border_dists[i]
                    if d > bd:
                        frac = np.arccos(min(1.0, bd / d)) / np.pi
                        weights[i, j] = 1.0 / (1.0 - frac)
                    else:
                        weights[i, j] = 1.0
    else:
        weights = np.ones(n)
        max_bd = border_dists.max()
        for i in range(n):
            weights[i] = max_bd / border_dists[i] if border_dists[i] > 0 else 1.0

    return DescriptiveResult(
        name="edge_correction",
        value=float(np.mean(border_dists)),
        extra={
            "weights": weights,
            "border_distances": border_dists.tolist(),
            "method": method,
            "mean_border_dist": float(np.mean(border_dists)),
        },
    )


sgedg = edge_correction


def cheatsheet() -> str:
    return "edge_correction({}) -> Edge correction weights for point patterns."
