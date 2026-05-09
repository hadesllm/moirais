# moirais.fn — function file (hadesllm/moirais)
"""Node betweenness centrality for a network."""

from __future__ import annotations

import numpy as np


def network_betweenness(
    adj_matrix: np.ndarray,
    *,
    node_names: list[str] | None = None,
) -> dict:
    """Node betweenness centrality via Brandes' algorithm on weighted graph.

    Betweenness counts the fraction of shortest paths passing through
    each node.  Edge weights are converted to distances as 1/|w|.

    Parameters
    ----------
    adj_matrix : ndarray
        Weighted adjacency matrix (p x p).
    node_names : list, optional
        Labels for nodes.

    Returns
    -------
    dict
        Keys: ``betweenness`` (dict), ``mean``, ``sd``.

    References
    ----------
    Brandes, U. (2001). A faster algorithm for betweenness centrality.
    *Journal of Mathematical Sociology*, 25(2), 163--177.
    """
    A = np.asarray(adj_matrix, dtype=np.float64)
    p = A.shape[0]
    names = node_names or [f"n{i}" for i in range(p)]
    np.fill_diagonal(A, 0.0)

    # Convert to distance: 1 / |weight|, inf for zero
    dist = np.full((p, p), np.inf)
    mask = A != 0
    dist[mask] = 1.0 / np.abs(A[mask])

    # Floyd-Warshall shortest paths + counting
    d = dist.copy()
    sigma = np.zeros((p, p))
    sigma[np.arange(p), np.arange(p)] = 1.0
    for e in range(p):
        if A[np.arange(p) != e].any() or True:
            pass  # just need adjacency

    # Simplified: use repeated Dijkstra via numpy
    betw = np.zeros(p)
    for s in range(p):
        # BFS/Dijkstra from s
        visited = np.zeros(p, dtype=bool)
        dist_s = np.full(p, np.inf)
        dist_s[s] = 0.0
        sig = np.zeros(p)
        sig[s] = 1.0
        pred = [[] for _ in range(p)]
        order = []

        for _ in range(p):
            # Find unvisited with smallest distance
            candidates = np.where(~visited)[0]
            if len(candidates) == 0:
                break
            u = candidates[np.argmin(dist_s[candidates])]
            if np.isinf(dist_s[u]):
                break
            visited[u] = True
            order.append(u)

            for v in range(p):
                if visited[v] or A[u, v] == 0:
                    continue
                w = 1.0 / abs(A[u, v])
                alt = dist_s[u] + w
                if alt < dist_s[v] - 1e-12:
                    dist_s[v] = alt
                    sig[v] = sig[u]
                    pred[v] = [u]
                elif abs(alt - dist_s[v]) < 1e-12:
                    sig[v] += sig[u]
                    pred[v].append(u)

        # Accumulate
        delta = np.zeros(p)
        for v in reversed(order):
            if v == s:
                continue
            for u in pred[v]:
                if sig[v] > 0:
                    delta[u] += (sig[u] / sig[v]) * (1.0 + delta[v])
            betw[v] += delta[v]

    # Normalize
    if p > 2:
        betw /= (p - 1) * (p - 2)

    result = {names[i]: float(betw[i]) for i in range(p)}
    return {
        "betweenness": result,
        "mean": float(np.mean(betw)),
        "sd": float(np.std(betw, ddof=1)) if p > 1 else 0.0,
    }


def cheatsheet() -> str:
    return "network_betweenness({}) -> Node betweenness centrality for a network."
