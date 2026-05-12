# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Binary search tree operations. 'The belonging you seek is ahead.' -- Maz Kanata"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bst_operations(
    values: np.ndarray,
    queries: np.ndarray | None = None,
) -> DescriptiveResult:
    """
    Build a BST from values and perform in-order traversal and search.

    :param values: Array of numeric values to insert.
    :param queries: Optional array of values to search for.
    :return: DescriptiveResult with traversal order, depth, and search results.

    References
    ----------
    Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
    (2009). *Introduction to Algorithms*. 3rd ed. MIT Press. Ch. 12.
    """
    vals = np.asarray(values, dtype=np.float64).ravel().tolist()
    if not vals:
        raise ValueError("Values array must be non-empty.")

    nodes: dict[int, dict] = {}
    root_idx = 0

    def _insert(val: float) -> None:
        idx = len(nodes)
        nodes[idx] = {"val": val, "left": -1, "right": -1}
        if idx == 0:
            return
        cur = 0
        while True:
            if val < nodes[cur]["val"]:
                if nodes[cur]["left"] == -1:
                    nodes[cur]["left"] = idx
                    return
                cur = nodes[cur]["left"]
            else:
                if nodes[cur]["right"] == -1:
                    nodes[cur]["right"] = idx
                    return
                cur = nodes[cur]["right"]

    for v in vals:
        _insert(v)

    inorder: list[float] = []

    def _inorder(idx: int) -> None:
        if idx == -1:
            return
        _inorder(nodes[idx]["left"])
        inorder.append(nodes[idx]["val"])
        _inorder(nodes[idx]["right"])

    _inorder(root_idx)

    def _depth(idx: int) -> int:
        if idx == -1:
            return 0
        return 1 + max(_depth(nodes[idx]["left"]), _depth(nodes[idx]["right"]))

    depth = _depth(root_idx)

    search_results = {}
    if queries is not None:
        q_arr = np.asarray(queries, dtype=np.float64).ravel().tolist()
        for q in q_arr:
            cur = root_idx
            found = False
            while cur != -1:
                if q == nodes[cur]["val"]:
                    found = True
                    break
                cur = nodes[cur]["left"] if q < nodes[cur]["val"] else nodes[cur]["right"]
            search_results[q] = found

    return DescriptiveResult(
        name="BST Operations",
        value=float(depth),
        extra={
            "inorder_traversal": inorder,
            "depth": depth,
            "n_nodes": len(nodes),
            "search_results": search_results,
        },
    )


short = bst_operations


def cheatsheet() -> str:
    return "bst_operations({}) -> Binary search tree operations. 'The belonging you seek is ah"
