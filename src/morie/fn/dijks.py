# morie.fn — function file (hadesllm/morie)
"""When I let go of what I am, I become what I might be. — Lao Tzu"""

from __future__ import annotations

import heapq

import numpy as np

from ._containers import DescriptiveResult


def dijkstra(
    adj_matrix: np.ndarray,
    source: int,
) -> DescriptiveResult:
    """
    Dijkstra's algorithm for single-source shortest paths.

    :param adj_matrix: Square adjacency/weight matrix (n x n). Zero or inf
        means no edge. Weights must be non-negative.
    :param source: Source node index (0-based).
    :return: DescriptiveResult with distances array as value.
    :raises ValueError: If matrix is not square, source out of range, or
        negative weights found.

    References
    ----------
    Dijkstra, E. W. (1959). A note on two problems in connexion with
    graphs. Numerische Mathematik, 1, 269--271. doi:10.1007/BF01386390
    """
    A = np.asarray(adj_matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adj_matrix must be a square 2-D array.")
    n = A.shape[0]
    if not 0 <= source < n:
        raise ValueError(f"source must be in [0, {n - 1}], got {source}.")
    if np.any(A[A != 0] < 0):
        raise ValueError("Negative weights are not supported by Dijkstra.")

    dist = np.full(n, np.inf)
    dist[source] = 0.0
    prev = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    heap = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        for v in range(n):
            w = A[u, v]
            if w > 0 and not visited[v]:
                alt = d + w
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(heap, (alt, v))

    return DescriptiveResult(
        name="Dijkstra Shortest Paths",
        value=dist,
        extra={
            "distances": dist,
            "predecessors": prev,
            "source": source,
            "n_nodes": n,
            "n_reachable": int(np.sum(np.isfinite(dist))),
        },
    )


dijks = dijkstra


def cheatsheet() -> str:
    return "When I let go of what I am, I become what I might be. — Lao Tzu"
