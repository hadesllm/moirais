# moirais.fn — function file (hadesllm/moirais)
"""K-means++ initialisation + Lloyd's algorithm."""

import numpy as np

from ._containers import DescriptiveResult


def kmeans_pp(X: np.ndarray, k: int = 3, n_iter: int = 100, seed: int = 42) -> DescriptiveResult:
    """
    K-means clustering with k-means++ initialisation (pure numpy).

    :param X: (n, p) data matrix.
    :param k: Number of clusters.
    :param n_iter: Maximum iterations.
    :param seed: Random seed.
    :return: DescriptiveResult with labels, centres, inertia.

    References
    ----------
    Arthur D, Vassilvitskii S (2007). k-means++: the advantages of
    careful seeding. SODA, 1027-1035.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    rng = np.random.default_rng(seed)
    centres = np.empty((k, p))
    centres[0] = X[rng.integers(n)]
    for c in range(1, k):
        dists = np.min([np.sum((X - centres[j]) ** 2, axis=1) for j in range(c)], axis=0)
        probs = dists / dists.sum()
        centres[c] = X[rng.choice(n, p=probs)]
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dists = np.array([np.sum((X - centres[j]) ** 2, axis=1) for j in range(k)])
        labels = np.argmin(dists, axis=0)
        new_centres = np.empty_like(centres)
        for j in range(k):
            mask = labels == j
            new_centres[j] = X[mask].mean(axis=0) if mask.any() else centres[j]
        if np.allclose(centres, new_centres):
            break
        centres = new_centres
    inertia = sum(np.sum((X[labels == j] - centres[j]) ** 2) for j in range(k))
    return DescriptiveResult(
        name="kmeans_pp",
        value=float(inertia),
        extra={"labels": labels, "centres": centres, "inertia": float(inertia), "k": k, "n": n},
    )


kmpp = kmeans_pp


def cheatsheet() -> str:
    return "kmeans_pp({}) -> K-means++ initialisation + Lloyd's algorithm."
