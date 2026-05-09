# moirais.fn — function file (hadesllm/moirais)
"""Network centrality measures. 'When I was a boy...' -- Kingpin"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def crime_network_centrality(
    adjacency: np.ndarray,
    *,
    directed: bool = False,
    alpha: float = 0.85,
) -> DescriptiveResult:
    """Compute multiple centrality measures for a crime/social network.

    Calculates degree centrality, betweenness centrality (Brandes approx),
    closeness centrality, and eigenvector centrality (power iteration).

    Parameters
    ----------
    adjacency : np.ndarray
        Square adjacency matrix (n x n).
    directed : bool
        Whether the graph is directed.
    alpha : float
        Damping factor for eigenvector centrality power iteration.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``degree``, ``betweenness``, ``closeness``,
        ``eigenvector`` (all n-length arrays), ``most_central`` (index).
    """
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency must be square")

    if not directed:
        A = (A + A.T) / 2

    deg = A.sum(axis=1)
    deg_cent = deg / (n - 1) if n > 1 else deg

    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)
    D[A > 0] = 1.0
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D[i, k] + D[k, j] < D[i, j]:
                    D[i, j] = D[i, k] + D[k, j]

    close = np.zeros(n)
    for i in range(n):
        reachable = D[i, :] < np.inf
        reachable[i] = False
        if reachable.sum() > 0:
            close[i] = reachable.sum() / D[i, reachable].sum()

    between = np.zeros(n)
    for s in range(n):
        for t in range(n):
            if s == t:
                continue
            if D[s, t] == np.inf:
                continue
            for v in range(n):
                if v == s or v == t:
                    continue
                if abs(D[s, v] + D[v, t] - D[s, t]) < 1e-10:
                    between[v] += 1.0
    norm = (n - 1) * (n - 2) if n > 2 else 1
    between /= norm

    x = np.ones(n) / n
    for _ in range(100):
        x_new = A @ x
        norm_val = np.linalg.norm(x_new)
        if norm_val > 0:
            x_new /= norm_val
        if np.linalg.norm(x_new - x) < 1e-8:
            break
        x = x_new
    eigvec = x

    most_central = int(np.argmax(eigvec))

    return DescriptiveResult(
        name="crime_network_centrality",
        value={
            "degree": deg_cent,
            "betweenness": between,
            "closeness": close,
            "eigenvector": eigvec,
            "most_central": most_central,
        },
        extra={"n": n, "directed": directed, "density": float(deg.sum()) / (n * (n - 1)) if n > 1 else 0},
    )


kngpn = crime_network_centrality


def cheatsheet() -> str:
    return "crime_network_centrality({}) -> Network centrality measures. 'When I was a boy...' -- Kingpi"
