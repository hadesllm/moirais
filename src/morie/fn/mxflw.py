# morie.fn — function file (hadesllm/morie)
"""He who has a why to live can bear almost any how. — Friedrich Nietzsche"""

from __future__ import annotations

from collections import deque

import numpy as np

from ._containers import DescriptiveResult


def max_flow(
    capacity: np.ndarray,
    source: int,
    sink: int,
) -> DescriptiveResult:
    """
    Compute maximum flow using Edmonds-Karp (BFS-based Ford-Fulkerson).

    :param capacity: Square capacity matrix (n x n). Zero means no edge.
    :param source: Source node index.
    :param sink: Sink node index.
    :return: DescriptiveResult with max flow value and residual graph.
    :raises ValueError: If matrix not square, source==sink, or indices OOB.

    References
    ----------
    Edmonds, J., & Karp, R. M. (1972). Theoretical improvements in
    algorithmic efficiency for network flow problems. Journal of the ACM,
    19(2), 248--264. doi:10.1145/321694.321699
    """
    C = np.array(capacity, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("capacity must be a square 2-D array.")
    n = C.shape[0]
    if source == sink:
        raise ValueError("source and sink must differ.")
    for name, val in [("source", source), ("sink", sink)]:
        if not 0 <= val < n:
            raise ValueError(f"{name} must be in [0, {n - 1}], got {val}.")

    residual = C.copy()
    total_flow = 0.0

    while True:
        parent = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        visited[source] = True
        queue = deque([source])
        found = False

        while queue and not found:
            u = queue.popleft()
            for v in range(n):
                if not visited[v] and residual[u, v] > 1e-12:
                    visited[v] = True
                    parent[v] = u
                    if v == sink:
                        found = True
                        break
                    queue.append(v)

        if not found:
            break

        path_flow = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual[u, v])
            v = u

        v = sink
        while v != source:
            u = parent[v]
            residual[u, v] -= path_flow
            residual[v, u] += path_flow
            v = u

        total_flow += path_flow

    return DescriptiveResult(
        name="Max Flow (Edmonds-Karp)",
        value=float(total_flow),
        extra={
            "max_flow": float(total_flow),
            "residual": residual,
            "source": source,
            "sink": sink,
            "n_nodes": n,
        },
    )


mxflw = max_flow


def cheatsheet() -> str:
    return "He who has a why to live can bear almost any how. — Friedrich Nietzsche"
