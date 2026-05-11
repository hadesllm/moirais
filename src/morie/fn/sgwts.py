"""Spatial weights matrix construction."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def spatial_weights_matrix(
    coords: np.ndarray,
    method: str = "knn",
    k: int = 5,
    bandwidth: float | None = None,
) -> SpatialResult:
    r"""Construct a spatial weights matrix.

    Parameters
    ----------
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    method : str
        ``"knn"`` or ``"distance"``.
    k : int
        Number of nearest neighbours (for knn).
    bandwidth : float, optional
        Distance threshold (for distance method).

    Returns
    -------
    SpatialResult
        ``statistic`` is average number of neighbours.
        ``extra`` has ``W`` (raw), ``W_row`` (row-standardized).

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 6.

    .. epigraph::

        "Keep your expectations low and you'll never be disappointed."
        -- Kratos, God of War
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    W = np.zeros((n, n))
    if method == "knn":
        for i in range(n):
            idx = np.argsort(dist[i])[1 : k + 1]
            W[i, idx] = 1.0
    elif method == "distance":
        bw = bandwidth or float(np.median(dist[dist > 0]))
        W = (dist <= bw).astype(np.float64)
        np.fill_diagonal(W, 0.0)
    else:
        raise ValueError(f"Unknown method: {method}")

    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W_row = W / rs
    avg_nn = float(np.mean(W.sum(axis=1)))

    return SpatialResult(
        name="spatial_weights_matrix",
        statistic=avg_nn,
        p_value=None,
        extra={"W": W, "W_row": W_row},
    )


sgwts = spatial_weights_matrix


def cheatsheet() -> str:
    return "spatial_weights_matrix({}) -> Spatial weights matrix construction."
