# morie.fn -- function file (hadesllm/morie)
"""Kruskal minimum spanning tree. 'We are the spark that will light the fire.' -- Poe Dameron"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def kruskal_mst(
    edges: np.ndarray,
    n_nodes: int,
) -> DescriptiveResult:
    """
    Kruskal's algorithm for minimum spanning tree.

    :param edges: Array of shape (m, 3) where each row is (u, v, weight).
        Nodes are 0-indexed integers.
    :param n_nodes: Total number of nodes.
    :return: DescriptiveResult with total MST weight and edge list.
    :raises ValueError: If edges has wrong shape or n_nodes < 2.

    References
    ----------
    Kruskal, J. B. (1956). On the shortest spanning subtree of a graph
    and the traveling salesman problem. Proceedings of the American
    Mathematical Society, 7(1), 48--50.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 2 or edges.shape[1] != 3:
        raise ValueError("edges must be shape (m, 3) with columns [u, v, weight].")
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2.")

    parent = list(range(n_nodes))
    rank = [0] * n_nodes

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    order = np.argsort(edges[:, 2])
    mst_edges = []
    total_weight = 0.0

    for idx in order:
        u, v, w = int(edges[idx, 0]), int(edges[idx, 1]), edges[idx, 2]
        if union(u, v):
            mst_edges.append((u, v, w))
            total_weight += w
            if len(mst_edges) == n_nodes - 1:
                break

    return DescriptiveResult(
        name="Kruskal MST",
        value=float(total_weight),
        extra={
            "total_weight": float(total_weight),
            "mst_edges": mst_edges,
            "n_edges_in_mst": len(mst_edges),
            "n_nodes": n_nodes,
            "is_connected": len(mst_edges) == n_nodes - 1,
        },
    )


krus = kruskal_mst


def cheatsheet() -> str:
    return "kruskal_mst({}) -> Kruskal minimum spanning tree. 'We are the spark that will l"
