"""DBSCAN density-based clustering (Ester et al. 1996)."""
import numpy as np

from ._richresult import RichResult

__all__ = ["dbscan_clustering"]


def dbscan_clustering(x, *, eps=0.5, min_samples=5, metric="euclidean"):
    """DBSCAN via sklearn.cluster.DBSCAN.

    Core points: |N_eps(x)| >= minPts.  Non-core points reachable from a
    core get the core's cluster label; otherwise they are noise (label -1).

    Parameters
    ----------
    x : array-like (n, p).
    eps : float
        Neighbourhood radius.
    min_samples : int
        Minimum size of an eps-neighbourhood for a core point.
    metric : str
        Distance metric (see sklearn).

    Returns
    -------
    RichResult with payload: estimate (number of clusters), labels,
    n_clusters, n_noise, core_sample_indices, n, method.
    """
    from sklearn.cluster import DBSCAN

    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = db.fit_predict(X)
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    n_noise = int(np.sum(labels == -1))
    return RichResult(payload={
        "estimate": n_clusters,
        "labels": labels.tolist(),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "core_sample_indices": db.core_sample_indices_.tolist(),
        "eps": float(eps),
        "min_samples": int(min_samples),
        "n": int(n),
        "method": "DBSCAN (Ester et al. 1996)",
    })


def cheatsheet():
    return "dbscl: DBSCAN density-based clustering"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    blob1 = rng.normal(loc=[0, 0], scale=0.3, size=(40, 2))
    blob2 = rng.normal(loc=[5, 5], scale=0.3, size=(40, 2))
    noise = rng.uniform(low=-2, high=7, size=(10, 2))
    X = np.vstack([blob1, blob2, noise])
    r = dbscan_clustering(X, eps=0.8, min_samples=5)
    print("n_clusters:", r.n_clusters, "  n_noise:", r.n_noise)
    print("labels (first 20):", r.labels[:20])
