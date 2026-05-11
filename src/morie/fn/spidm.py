"""He who is brave is free. — Seneca"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def web_graph_communities(
    adjacency: np.ndarray,
    *,
    k: int = 4,
    n_iter: int = 50,
    seed: int | None = None,
) -> DescriptiveResult:
    """Detect communities in a graph using spectral partitioning on the
    normalised Laplacian.

    Computes the first *k* eigenvectors of the symmetric normalised graph
    Laplacian :math:`L_{sym} = I - D^{-1/2} A D^{-1/2}` and clusters rows
    with k-means.

    Parameters
    ----------
    adjacency : np.ndarray
        Square symmetric adjacency matrix (n x n).
    k : int
        Number of communities to detect.
    n_iter : int
        Max k-means iterations.
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    DescriptiveResult
        ``value`` is a dict with ``labels`` (n,) and ``modularity`` float.
    """
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    if k < 2 or k > n:
        raise ValueError(f"k must be in [2, {n}]")

    deg = A.sum(axis=1)
    deg[deg == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(L_sym)
    U = eigvecs[:, :k]
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U = U / norms

    rng = np.random.default_rng(seed)
    centers = U[rng.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dists = np.array([np.linalg.norm(U - c, axis=1) for c in centers])
        new_labels = dists.argmin(axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = U[mask].mean(axis=0)

    m = A.sum() / 2
    Q = 0.0
    if m > 0:
        for i in range(n):
            for j in range(n):
                if labels[i] == labels[j]:
                    Q += A[i, j] - deg[i] * deg[j] / (2 * m)
        Q /= 2 * m

    return DescriptiveResult(
        name="web_graph_communities",
        value={"labels": labels, "modularity": float(Q)},
        extra={"k": k, "n": n},
    )


spidm = web_graph_communities


def cheatsheet() -> str:
    return "web_graph_communities({}) -> Web graph community detection. 'With great power comes great"
