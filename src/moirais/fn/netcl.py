# moirais.fn — function file (hadesllm/moirais)
"""Node closeness centrality for a network."""

from __future__ import annotations

import numpy as np


def network_closeness(
    adj_matrix: np.ndarray,
    *,
    node_names: list[str] | None = None,
) -> dict:
    """Node closeness centrality (inverse of average shortest path).

    Edge weights are treated as distances via 1/|w|.  Disconnected
    nodes receive closeness = 0.

    Parameters
    ----------
    adj_matrix : ndarray
        Weighted adjacency matrix (p x p).
    node_names : list, optional
        Node labels.

    Returns
    -------
    dict
        Keys: ``closeness`` (dict), ``mean``, ``sd``.

    References
    ----------
    Opsahl, T., Agneessens, F., & Skvoretz, J. (2010). Node centrality
    in weighted networks. *Social Networks*, 32(3), 245--251.
    """
    A = np.asarray(adj_matrix, dtype=np.float64)
    p = A.shape[0]
    names = node_names or [f"n{i}" for i in range(p)]
    np.fill_diagonal(A, 0.0)

    # Distance matrix via Floyd-Warshall
    dist = np.full((p, p), np.inf)
    np.fill_diagonal(dist, 0.0)
    for i in range(p):
        for j in range(p):
            if i != j and A[i, j] != 0:
                dist[i, j] = 1.0 / abs(A[i, j])

    for k in range(p):
        for i in range(p):
            for j in range(p):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    close = np.zeros(p)
    for i in range(p):
        reachable = dist[i, :] < np.inf
        reachable[i] = False
        n_reach = np.sum(reachable)
        if n_reach > 0:
            close[i] = n_reach / np.sum(dist[i, reachable])

    result = {names[i]: float(close[i]) for i in range(p)}
    return {
        "closeness": result,
        "mean": float(np.mean(close)),
        "sd": float(np.std(close, ddof=1)) if p > 1 else 0.0,
    }


def cheatsheet() -> str:
    return "network_closeness({}) -> Node closeness centrality for a network."
