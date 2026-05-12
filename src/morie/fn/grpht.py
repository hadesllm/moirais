# morie.fn -- function file (hadesllm/morie)
"""Graph from edge list. 'We are what they grow beyond.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def graph_from_edges(
    edges: list[tuple],
    directed: bool = False,
) -> DescriptiveResult:
    """
    Build an adjacency representation from an edge list.

    Returns adjacency list, adjacency matrix, and basic graph stats
    (degree distribution, connected components for undirected).

    :param edges: List of (source, target) tuples. Optional third
        element is edge weight (default 1.0).
    :param directed: If True, edges are directed. Default False.
    :return: DescriptiveResult with adjacency matrix and graph stats.
    :raises ValueError: If edges list is empty.

    References
    ----------
    Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
    (2009). *Introduction to Algorithms*. 3rd ed. MIT Press. Ch. 22.
    """
    if not edges:
        raise ValueError("Edge list must be non-empty.")

    nodes_set: set = set()
    for e in edges:
        nodes_set.add(e[0])
        nodes_set.add(e[1])
    nodes = sorted(nodes_set)
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    adj_matrix = np.zeros((n, n), dtype=np.float64)
    adj_list: dict = {node: [] for node in nodes}

    for e in edges:
        w = e[2] if len(e) > 2 else 1.0
        i, j = node_idx[e[0]], node_idx[e[1]]
        adj_matrix[i, j] = w
        adj_list[e[0]].append(e[1])
        if not directed:
            adj_matrix[j, i] = w
            adj_list[e[1]].append(e[0])

    if directed:
        in_degree = np.sum(adj_matrix > 0, axis=0).astype(int)
        out_degree = np.sum(adj_matrix > 0, axis=1).astype(int)
        degree_info = {"in_degree": in_degree, "out_degree": out_degree}
    else:
        degree = np.sum(adj_matrix > 0, axis=1).astype(int)
        degree_info = {"degree": degree}

    n_components = 0
    if not directed:
        visited = set()

        def _bfs(start: int) -> None:
            queue = [start]
            while queue:
                cur = queue.pop(0)
                for nb in range(n):
                    if adj_matrix[cur, nb] > 0 and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)

        for i in range(n):
            if i not in visited:
                visited.add(i)
                _bfs(i)
                n_components += 1

    return DescriptiveResult(
        name="Graph",
        value=float(n),
        extra={
            "adj_matrix": adj_matrix,
            "nodes": nodes,
            "n_nodes": n,
            "n_edges": len(edges),
            "directed": directed,
            "density": float(np.sum(adj_matrix > 0)) / (n * (n - 1)) if n > 1 else 0.0,
            "n_components": n_components if not directed else None,
            **degree_info,
        },
    )


short = graph_from_edges


def cheatsheet() -> str:
    return "graph_from_edges({}) -> Graph from edge list. 'We are what they grow beyond.' -- Yod"
