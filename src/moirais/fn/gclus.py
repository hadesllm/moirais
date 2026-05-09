# moirais.fn — function file (hadesllm/moirais)
"""Geographic k-means clustering. 'Rebellions are built on hope.' -- Jyn Erso"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def geo_cluster(
    lat: np.ndarray,
    lon: np.ndarray,
    n_clusters: int = 5,
    max_iter: int = 100,
    seed: int = 42,
) -> DescriptiveResult:
    """K-means clustering on geographic coordinates.

    Converts lat/lon to 3D Cartesian (unit sphere), runs k-means,
    and converts centroids back to lat/lon. Handles the wraparound
    problem inherent in angular coordinates.

    Parameters
    ----------
    lat : ndarray, shape (n_points,)
        Latitudes in decimal degrees.
    lon : ndarray, shape (n_points,)
        Longitudes in decimal degrees.
    n_clusters : int
        Number of clusters.
    max_iter : int
        Maximum k-means iterations.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        name='Geo Cluster', value=inertia,
        extra has 'labels' (ndarray), 'centroids_lat', 'centroids_lon',
        'n_clusters', 'n_iter'.

    References
    ----------
    MacQueen, J. (1967). Some methods for classification and analysis
    of multivariate observations. *Proceedings of 5th Berkeley Symposium
    on Mathematical Statistics and Probability*, 1, 281-297.
    """
    rng = np.random.default_rng(seed)
    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon = np.asarray(lon, dtype=np.float64).ravel()
    n = len(lat)

    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    X = np.column_stack([x, y, z])

    init_idx = rng.choice(n, size=min(n_clusters, n), replace=False)
    centroids = X[init_idx].copy()

    labels = np.zeros(n, dtype=int)
    for it in range(1, max_iter + 1):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels) and it > 1:
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                c = X[mask].mean(axis=0)
                norm = np.linalg.norm(c)
                centroids[k] = c / norm if norm > 0 else c

    c_lat = np.rad2deg(np.arcsin(np.clip(centroids[:, 2], -1, 1)))
    c_lon = np.rad2deg(np.arctan2(centroids[:, 1], centroids[:, 0]))

    inertia = 0.0
    for k in range(n_clusters):
        mask = labels == k
        if mask.any():
            inertia += float(np.sum(np.linalg.norm(X[mask] - centroids[k], axis=1) ** 2))

    return DescriptiveResult(
        name="Geo Cluster",
        value=float(inertia),
        extra={
            "labels": labels,
            "centroids_lat": c_lat,
            "centroids_lon": c_lon,
            "n_clusters": n_clusters,
            "n_iter": it,
            "n_points": n,
        },
    )


gclus = geo_cluster


def cheatsheet() -> str:
    return "geo_cluster({}) -> Geographic k-means clustering. 'Rebellions are built on hope"
