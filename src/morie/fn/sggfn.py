"""G function (nearest-neighbor CDF)."""

from __future__ import annotations

from ._containers import DescriptiveResult


def g_function_nearest_neighbor(points):
    """Compute G(r), the nearest-neighbor distance CDF.

    G(r) above CSR envelope indicates clustering.

    .. epigraph:: I think, therefore I am. -- Rene Descartes

    Parameters
    ----------
    points : array_like
        Point coordinates, shape ``(n, 2)``.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]

    D = squareform(pdist(pts))
    np.fill_diagonal(D, np.inf)
    nn_dists = np.sort(D.min(axis=1))

    r_values = np.linspace(0, nn_dists.max(), 50)
    G = np.array([np.mean(nn_dists <= r) for r in r_values])

    area_est = (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
    lam = n / area_est if area_est > 0 else 1.0
    G_csr = 1 - np.exp(-lam * np.pi * r_values**2)

    return DescriptiveResult(
        name="g_function_nearest_neighbor",
        value=float(nn_dists.mean()),
        extra={
            "r_values": r_values.tolist(),
            "G_values": G.tolist(),
            "G_csr": G_csr.tolist(),
            "mean_nn_distance": float(nn_dists.mean()),
            "nn_distances": nn_dists.tolist(),
        },
    )


sggfn = g_function_nearest_neighbor


def cheatsheet() -> str:
    return "g_function_nearest_neighbor({}) -> G function (nearest-neighbor CDF)."
