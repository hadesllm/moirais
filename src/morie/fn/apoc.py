# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Graph articulation points. 'I am afraid of the truth.' -- Apoc"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def articulation_points(
    adj: np.ndarray | list[list[int]],
) -> DescriptiveResult:
    """Find articulation points (cut vertices) in an undirected graph.

    Uses Tarjan's DFS algorithm. An articulation point is a vertex whose
    removal disconnects the graph.

    Parameters
    ----------
    adj : ndarray or list of lists
        Adjacency matrix (n x n), symmetric, with 0/1 entries.

    Returns
    -------
    DescriptiveResult
        ``value`` is the number of articulation points; ``extra`` has the
        list of articulation point indices and component count.
    """
    A = np.asarray(adj, dtype=int)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0:
                neighbors[i].append(j)
                neighbors[j].append(i)

    visited = [False] * n
    disc = [0] * n
    low = [0] * n
    parent = [-1] * n
    ap_set = set()
    timer = [0]

    def _dfs(u):
        children = 0
        visited[u] = True
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in neighbors[u]:
            if not visited[v]:
                children += 1
                parent[v] = u
                _dfs(v)
                low[u] = min(low[u], low[v])
                if parent[u] == -1 and children > 1:
                    ap_set.add(u)
                if parent[u] != -1 and low[v] >= disc[u]:
                    ap_set.add(u)
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    components = 0
    for i in range(n):
        if not visited[i]:
            _dfs(i)
            components += 1

    ap_list = sorted(ap_set)
    return DescriptiveResult(
        name="Articulation Points",
        value=len(ap_list),
        extra={
            "points": ap_list,
            "n_vertices": n,
            "n_components": components,
            "n_edges": int(A.sum()) // 2,
        },
    )


apoc = articulation_points


def cheatsheet() -> str:
    return "articulation_points({}) -> Graph articulation points. 'I am afraid of the truth.' -- Ap"
