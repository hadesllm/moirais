# morie.fn — function file (hadesllm/morie)
"""Network average shortest path length (BFS)."""

from __future__ import annotations

from collections import deque

import numpy as np

from ._containers import ESRes


def network_path_length(adj: np.ndarray) -> ESRes:
    """Average shortest path length via BFS.

    Parameters
    ----------
    adj : (n, n) binary adjacency matrix

    Returns
    -------
    ESRes
    """
    A = (np.asarray(adj, dtype=float) > 0).astype(int)
    np.fill_diagonal(A, 0)
    n = A.shape[0]

    total_dist = 0
    n_pairs = 0
    for src in range(n):
        dist = -np.ones(n, dtype=int)
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in range(n):
                if A[u, v] and dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        reachable = dist[dist > 0]
        total_dist += reachable.sum()
        n_pairs += len(reachable)

    avg_path = total_dist / n_pairs if n_pairs > 0 else float("inf")

    return ESRes(
        measure="avg_path_length",
        estimate=float(avg_path),
        n=n,
        extra={"n_reachable_pairs": n_pairs},
    )


netpl = network_path_length


def cheatsheet() -> str:
    return "network_path_length({}) -> Network average shortest path length (BFS)."
