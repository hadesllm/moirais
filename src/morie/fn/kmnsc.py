"""K-means clustering via Lloyd's algorithm."""
import numpy as np

from ._richresult import RichResult

__all__ = ["kmeans_clustering"]


def kmeans_clustering(x, *, n_clusters=3, n_init=10, max_iter=300, seed=0):
    """K-means via sklearn.cluster.KMeans (Lloyd's algorithm, k-means++ init).

    Minimises sum_i ||x_i - mu_{k(i)}||^2 by alternating assignment and
    centroid update steps.

    Parameters
    ----------
    x : array-like (n, p).
    n_clusters : int
        Number of clusters K.
    n_init : int
        Number of random k-means++ restarts.
    max_iter : int
        Max Lloyd-iterations per restart.
    seed : int
        random_state.

    Returns
    -------
    RichResult with payload: estimate (inertia / WCSS), labels, centers
    (n_clusters x p), inertia, n_iter, n, method.
    """
    from sklearn.cluster import KMeans

    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    km = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter,
                random_state=seed)
    km.fit(X)
    return RichResult(payload={
        "estimate": float(km.inertia_),
        "labels": km.labels_.tolist(),
        "centers": km.cluster_centers_.tolist(),
        "inertia": float(km.inertia_),
        "n_iter": int(km.n_iter_),
        "n_clusters": int(n_clusters),
        "n": int(n),
        "method": "K-means (Lloyd, k-means++ init)",
    })


def cheatsheet():
    return "kmnsc: k-means clustering (Lloyd's algorithm)"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [5, 5], [-5, 5]])
    X = np.vstack([rng.normal(loc=c, size=(40, 2)) for c in centres])
    r = kmeans_clustering(X, n_clusters=3, seed=0)
    print("inertia:", r.inertia)
    print("n_iter:", r.n_iter)
    print("centres:", r.centers)
