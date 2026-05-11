# morie.fn — function file (hadesllm/morie)
"""Connected components via BFS."""

from __future__ import annotations

from collections import deque

import numpy as np

from ._containers import DescriptiveResult


def connected_components(adj_matrix: np.ndarray) -> DescriptiveResult:
    """Find connected components in an undirected graph using BFS.

    Parameters
    ----------
    adj_matrix : ndarray
        Square adjacency matrix (n x n).  Treated as undirected: an
        edge exists between *i* and *j* if ``A[i,j] != 0`` or
        ``A[j,i] != 0``.

    Returns
    -------
    DescriptiveResult
        ``value`` is the number of connected components.
        ``extra`` has ``labels`` (1-D int array mapping each node to
        its component index) and ``sizes`` (list of component sizes,
        sorted descending).

    References
    ----------
    Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
    (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.
    Ch. 22: BFS.
    """
    A = np.asarray(adj_matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adj_matrix must be square.")

    A_sym = ((A != 0) | (A.T != 0)).astype(np.float64)
    np.fill_diagonal(A_sym, 0)
    n = A_sym.shape[0]

    labels = np.full(n, -1, dtype=int)
    comp_id = 0

    for start in range(n):
        if labels[start] >= 0:
            continue
        queue: deque[int] = deque([start])
        labels[start] = comp_id
        while queue:
            v = queue.popleft()
            for w in range(n):
                if A_sym[v, w] != 0 and labels[w] < 0:
                    labels[w] = comp_id
                    queue.append(w)
        comp_id += 1

    sizes = sorted([int(np.sum(labels == c)) for c in range(comp_id)], reverse=True)

    return DescriptiveResult(
        name="ConnectedComponents",
        value=comp_id,
        extra={"labels": labels, "sizes": sizes, "n_nodes": n},
    )


cmpnt = connected_components


def cheatsheet() -> str:
    return "connected_components({}) -> Connected components via BFS."
