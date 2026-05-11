"""Spatial weight matrix construction."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def spatial_weights(
    coords: np.ndarray,
    method: str = "knn",
    k: int = 5,
    threshold: float | None = None,
    row_standardize: bool = True,
) -> SpatialResult:
    r"""
    Construct a spatial weight matrix from point coordinates.

    Supports three methods:

    - **knn**: :math:`w_{ij} = 1` if *j* is among the *k* nearest neighbors
      of *i*, 0 otherwise.
    - **distance**: :math:`w_{ij} = 1` if :math:`d_{ij} \le \text{threshold}`,
      0 otherwise.
    - **inverse**: :math:`w_{ij} = 1/d_{ij}` if :math:`d_{ij} \le \text{threshold}`.

    Self-neighbors (:math:`w_{ii}`) are always zero. If ``row_standardize``
    is True, each row sums to 1.

    Parameters
    ----------
    coords : np.ndarray
        (n, 2) spatial coordinates.
    method : str
        ``"knn"``, ``"distance"``, or ``"inverse"`` (default ``"knn"``).
    k : int
        Number of neighbors for knn (default 5).
    threshold : float or None
        Distance threshold for ``"distance"``/``"inverse"`` methods.
        If None, uses the median pairwise distance.
    row_standardize : bool
        Whether to row-standardize W (default True).

    Returns
    -------
    SpatialResult
        statistic = mean number of neighbors per unit, extra has ``W``
        (n, n) weight matrix and ``n_islands`` (units with no neighbors).

    References
    ----------
    Anselin L (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
    Chapter 3.

    Getis A, Aldstadt J (2004). Constructing the spatial weights matrix
    using a local statistic. *Geographical Analysis*, 36(2), 90--104.
    doi:10.1111/j.1538-4632.2004.tb01127.x

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.array([[0,0],[1,0],[0,1],[1,1],[0.5,0.5]], dtype=float)
    >>> res = spatial_weights(coords, method="knn", k=2)
    >>> res.extra["W"].shape == (5, 5)
    True
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be (n, 2).")
    n = coords.shape[0]

    dists = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)

    W = np.zeros((n, n))

    if method == "knn":
        for i in range(n):
            idx = np.argpartition(dists[i], k)[:k]
            W[i, idx] = 1.0
    elif method == "distance":
        if threshold is None:
            finite_dists = dists[np.isfinite(dists)]
            threshold = float(np.median(finite_dists))
        W = (dists <= threshold).astype(float)
    elif method == "inverse":
        if threshold is None:
            finite_dists = dists[np.isfinite(dists)]
            threshold = float(np.median(finite_dists))
        mask = dists <= threshold
        safe_dists = np.where(dists < 1e-15, 1e-15, dists)
        W = np.where(mask, 1.0 / safe_dists, 0.0)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'knn', 'distance', or 'inverse'.")

    np.fill_diagonal(W, 0.0)

    n_islands = int(np.sum(W.sum(axis=1) == 0))

    if row_standardize:
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        W = W / row_sums

    mean_neighbors = float(np.mean(np.sum(W > 0, axis=1)))

    return SpatialResult(
        name="spatial_weights",
        statistic=mean_neighbors,
        extra={"W": W, "method": method, "n": n, "n_islands": n_islands, "row_standardized": row_standardize},
    )


spwgt = spatial_weights


def cheatsheet() -> str:
    return "spatial_weights({}) -> Spatial weight matrix construction."
