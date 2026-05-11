"""ST-DBSCAN spatiotemporal clustering (Birant & Kut 2007)."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def st_dbscan(
    coords: np.ndarray,
    times: np.ndarray,
    eps_spatial: float,
    eps_temporal: float,
    min_pts: int = 5,
) -> SpatialResult:
    r"""ST-DBSCAN: density-based spatiotemporal clustering.

    Extends DBSCAN by defining the neighborhood of point *i* as all
    points *j* satisfying both:

    .. math::

        \|s_i - s_j\| \le \varepsilon_s
        \quad\text{and}\quad
        |t_i - t_j| \le \varepsilon_t

    Core points have at least ``min_pts`` neighbors. Clusters are
    formed by density-reachability in the joint (space, time) metric.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates, shape ``(n, 2)``.
    times : np.ndarray
        Time values, shape ``(n,)``.
    eps_spatial : float
        Spatial neighborhood radius :math:`\varepsilon_s`.
    eps_temporal : float
        Temporal neighborhood radius :math:`\varepsilon_t`.
    min_pts : int
        Minimum neighbors for a core point. Default 5.

    Returns
    -------
    SpatialResult
        ``statistic`` is the number of clusters found.
        ``extra`` contains ``labels`` (-1 = noise), ``n_noise``,
        ``cluster_sizes``.

    References
    ----------
    Birant D, Kut A (2007). ST-DBSCAN: An algorithm for clustering
    spatial-temporal data. *Data & Knowledge Engineering*, 60(1),
    208-221.

    Ester M, Kriegel H-P, Sander J, Xu X (1996). A density-based
    algorithm for discovering clusters in large spatial databases with
    noise. *Proceedings of KDD*, 226-231.
    """
    xy = np.asarray(coords, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64).ravel()
    n = len(t)

    if xy.shape != (n, 2):
        raise ValueError("coords must be (n, 2)")

    sdist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
    tdist = np.abs(t[:, None] - t[None, :])

    neighbor_mask = (sdist <= eps_spatial) & (tdist <= eps_temporal)
    np.fill_diagonal(neighbor_mask, False)

    neighbors = [np.where(neighbor_mask[i])[0] for i in range(n)]
    n_neighbors = np.array([len(nb) for nb in neighbors])
    is_core = n_neighbors >= min_pts

    labels = np.full(n, -1, dtype=int)
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1 or not is_core[i]:
            continue
        queue = [i]
        labels[i] = cluster_id
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            for nb in neighbors[current]:
                if labels[nb] == -1:
                    labels[nb] = cluster_id
                    if is_core[nb]:
                        queue.append(nb)
        cluster_id += 1

    n_noise = int(np.sum(labels == -1))
    cluster_sizes = {}
    for cid in range(cluster_id):
        cluster_sizes[cid] = int(np.sum(labels == cid))

    return SpatialResult(
        name="ST_DBSCAN",
        statistic=float(cluster_id),
        p_value=None,
        extra={
            "labels": labels,
            "n_noise": n_noise,
            "n_clusters": cluster_id,
            "cluster_sizes": cluster_sizes,
        },
    )


def cheatsheet() -> str:
    return "st_dbscan({}) -> ST-DBSCAN spatiotemporal clustering (Birant & Kut 2007)."
