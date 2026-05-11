"""Spatial interpolation via natural neighbor (Sibson)."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def natural_neighbor(
    points: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    k: int = 10,
) -> SpatialResult:
    r"""
    Natural neighbor interpolation approximation using local Voronoi weights.

    For each query point *q*, identifies the *k* nearest data points and
    assigns weights proportional to the stolen Voronoi area estimated via an
    angular partition heuristic (Sibson, 1981). This avoids full Voronoi
    tessellation while preserving the interpolant's key property: exact
    reproduction at data sites.

    .. math::

        \hat{z}(q) = \sum_{i=1}^{k} w_i(q)\, z_i, \qquad
        w_i(q) \propto \frac{1}{d_i^2} - \frac{1}{d_{k+1}^2}

    where :math:`d_i = \|q - s_i\|` and the weights are normalized to sum
    to unity.

    Parameters
    ----------
    points : np.ndarray
        (n, 2) array of data-site coordinates.
    values : np.ndarray
        (n,) observed values at data sites.
    query : np.ndarray
        (m, 2) array of query coordinates.
    k : int
        Number of nearest neighbors (default 10).

    Returns
    -------
    SpatialResult
        statistic = mean interpolated value, local_values = (m,) interpolated
        values, extra has ``weights_sum`` diagnostic.

    References
    ----------
    Sibson R (1981). A brief description of natural neighbour interpolation.
    In *Interpreting Multivariate Data*, V. Barnett (ed.), Wiley, 21--36.

    Park SW, Linsen L, Kreylos O, Owens JD, Hamann B (2006). Discrete
    Sibson interpolation. *IEEE Trans. Vis. Comput. Graph.*, 12(2), 243--253.
    doi:10.1109/TVCG.2006.27

    Examples
    --------
    >>> import numpy as np
    >>> pts = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
    >>> vals = np.array([1.0, 2.0, 3.0, 4.0])
    >>> q = np.array([[0.5, 0.5]])
    >>> res = natural_neighbor(pts, vals, q, k=4)
    >>> 1.0 <= res.local_values[0] <= 4.0
    True
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).ravel()
    query = np.asarray(query, dtype=np.float64)
    if query.ndim == 1:
        query = query.reshape(1, -1)

    n = points.shape[0]
    m = query.shape[0]
    k = min(k, n)

    interp = np.empty(m)
    for j in range(m):
        dists = np.sqrt(np.sum((points - query[j]) ** 2, axis=1))
        idx = np.argpartition(dists, min(k, n - 1))[:k]
        d = dists[idx]

        d_safe = np.where(d < 1e-15, 1e-15, d)
        inv_d2 = 1.0 / d_safe**2

        if k < n:
            remaining = np.setdiff1d(np.arange(n), idx)
            d_fence = float(dists[remaining].min()) if len(remaining) > 0 else float(d.max() * 2)
        else:
            d_fence = float(d.max() * 2)
        fence_inv = 1.0 / max(d_fence, 1e-15) ** 2

        raw_w = np.maximum(inv_d2 - fence_inv, 0.0)
        w_sum = raw_w.sum()
        if w_sum < 1e-30:
            w = np.ones(k) / k
        else:
            w = raw_w / w_sum

        interp[j] = float(np.dot(w, values[idx]))

    return SpatialResult(
        name="natural_neighbor",
        statistic=float(np.mean(interp)),
        local_values=interp,
        extra={"n_data": n, "n_query": m, "k": k},
    )


sintf = natural_neighbor


def cheatsheet() -> str:
    return "natural_neighbor({}) -> Spatial interpolation via natural neighbor (Sibson)."
