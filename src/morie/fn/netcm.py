# morie.fn — function file (hadesllm/morie)
"""Community detection in a network."""

from __future__ import annotations

import numpy as np


def network_communities(
    adj_matrix: np.ndarray,
    *,
    method: str = "walktrap",
    n_communities: int | None = None,
    node_names: list[str] | None = None,
) -> dict:
    """Community detection via spectral clustering or greedy modularity.

    Parameters
    ----------
    adj_matrix : ndarray
        Weighted adjacency matrix (p x p).
    method : str
        ``'walktrap'`` (spectral, default) or ``'modularity'`` (greedy).
    n_communities : int, optional
        Number of communities.  If *None*, determined automatically via
        the eigengap heuristic.
    node_names : list, optional
        Node labels.

    Returns
    -------
    dict
        Keys: ``labels`` (dict node->community), ``n_communities``,
        ``modularity``, ``sizes``.

    References
    ----------
    Pons, P., & Latapy, M. (2005). Computing communities in large
    networks using random walks. *JGAA*, 10(2), 191--218.
    """
    A = np.asarray(adj_matrix, dtype=np.float64)
    p = A.shape[0]
    names = node_names or [f"n{i}" for i in range(p)]
    np.fill_diagonal(A, 0.0)
    A_abs = np.abs(A)

    # Degree matrix
    deg = np.sum(A_abs, axis=1)
    deg[deg == 0] = 1e-10

    if method == "walktrap":
        # Spectral clustering on normalized Laplacian
        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        L_norm = np.eye(p) - D_inv_sqrt @ A_abs @ D_inv_sqrt
        eigvals, eigvecs = np.linalg.eigh(L_norm)

        if n_communities is None:
            # Eigengap heuristic
            gaps = np.diff(eigvals[: min(p, 10)])
            n_communities = int(np.argmax(gaps) + 1)
            n_communities = max(2, min(n_communities, p))

        # Use first k eigenvectors
        V = eigvecs[:, :n_communities]
        # Normalize rows
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        V = V / norms

        # Simple k-means (Lloyd's algorithm)
        rng = np.random.default_rng(42)
        centers = V[rng.choice(p, n_communities, replace=False)]
        for _ in range(50):
            dists = np.array([np.sum((V - c) ** 2, axis=1) for c in centers])
            labels = np.argmin(dists, axis=0)
            for k in range(n_communities):
                members = V[labels == k]
                if len(members) > 0:
                    centers[k] = members.mean(axis=0)
    else:
        # Greedy modularity (simplified)
        labels = np.arange(p)
        m = np.sum(A_abs) / 2.0
        if m == 0:
            labels = np.zeros(p, dtype=int)
        else:
            improved = True
            while improved:
                improved = False
                for i in range(p):
                    best_gain = 0.0
                    best_comm = labels[i]
                    for c in np.unique(labels):
                        if c == labels[i]:
                            continue
                        # Modularity gain of moving i to c
                        ki = deg[i]
                        in_c = np.sum(A_abs[i, labels == c])
                        sum_c = np.sum(deg[labels == c])
                        gain = (in_c / m) - (ki * sum_c) / (2 * m**2)
                        if gain > best_gain:
                            best_gain = gain
                            best_comm = c
                    if best_comm != labels[i]:
                        labels[i] = best_comm
                        improved = True

        # Relabel contiguously
        unique = np.unique(labels)
        remap = {old: new for new, old in enumerate(unique)}
        labels = np.array([remap[l] for l in labels])
        n_communities = len(unique)

    # Compute modularity
    m = np.sum(A_abs) / 2.0
    Q = 0.0
    if m > 0:
        for i in range(p):
            for j in range(p):
                if labels[i] == labels[j]:
                    Q += A_abs[i, j] - deg[i] * deg[j] / (2 * m)
        Q /= 2 * m

    label_dict = {names[i]: int(labels[i]) for i in range(p)}
    sizes = {}
    for c in range(n_communities):
        sizes[c] = int(np.sum(labels == c))

    return {
        "labels": label_dict,
        "n_communities": n_communities,
        "modularity": float(Q),
        "sizes": sizes,
        "method": method,
    }


def cheatsheet() -> str:
    return "network_communities({}) -> Community detection in a network."
