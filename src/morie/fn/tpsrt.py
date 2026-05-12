"""Our greatest glory is not in never falling, but in rising every time we fall. -- Confucius"""

from __future__ import annotations

from collections import deque

from ._containers import DescriptiveResult


def topological_sort(adj_list: dict[int, list[int]]) -> DescriptiveResult:
    """
    Kahn's algorithm for topological ordering of a directed acyclic graph.

    :param adj_list: Adjacency list ``{node: [successors]}``.
    :return: DescriptiveResult with topological ordering.
    :raises ValueError: If graph contains a cycle.

    References
    ----------
    Kahn, A. B. (1962). Topological sorting of large networks.
    Communications of the ACM, 5(11), 558--562. doi:10.1145/368996.369025
    """
    if not adj_list:
        raise ValueError("adj_list must be non-empty.")

    nodes = set(adj_list.keys())
    for succs in adj_list.values():
        nodes.update(succs)

    in_degree = {n: 0 for n in nodes}
    for u, succs in adj_list.items():
        for v in succs:
            in_degree[v] = in_degree.get(v, 0) + 1

    queue = deque(n for n in sorted(nodes) if in_degree[n] == 0)
    order = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adj_list.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(order) != len(nodes):
        raise ValueError("Graph contains a cycle; topological sort undefined.")

    return DescriptiveResult(
        name="Topological Sort",
        value=order,
        extra={
            "order": order,
            "n_nodes": len(nodes),
            "n_edges": sum(len(v) for v in adj_list.values()),
        },
    )


tpsrt = topological_sort


def cheatsheet() -> str:
    return "topological_sort({}) -> Topological sort of a DAG. 'Your eyes can deceive you; don't"
