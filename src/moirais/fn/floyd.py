# moirais.fn — function file (hadesllm/moirais)
"""Floyd-Warshall all-pairs shortest paths. 'I have spoken.' -- Kuiil"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def floyd_warshall(dist_matrix: np.ndarray) -> DescriptiveResult:
    """
    Floyd-Warshall algorithm for all-pairs shortest paths.

    :param dist_matrix: Square distance matrix (n x n). Use ``np.inf``
        for non-edges. Diagonal should be 0.
    :return: DescriptiveResult with shortest-path distance matrix.
    :raises ValueError: If matrix is not square.

    References
    ----------
    Floyd, R. W. (1962). Algorithm 97: Shortest Path. Communications
    of the ACM, 5(6), 345. doi:10.1145/367766.368168
    """
    D = np.array(dist_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("dist_matrix must be a square 2-D array.")

    n = D.shape[0]
    nxt = np.full((n, n), -1, dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(D[i, j]):
                nxt[i, j] = j

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D[i, k] + D[k, j] < D[i, j]:
                    D[i, j] = D[i, k] + D[k, j]
                    nxt[i, j] = nxt[i, k]

    has_neg_cycle = any(D[i, i] < 0 for i in range(n))

    return DescriptiveResult(
        name="Floyd-Warshall",
        value=D,
        extra={
            "distance_matrix": D,
            "next_matrix": nxt,
            "n_nodes": n,
            "has_negative_cycle": has_neg_cycle,
            "diameter": float(np.max(D[np.isfinite(D)])) if np.any(np.isfinite(D)) else float("inf"),
        },
    )


floyd = floyd_warshall


def cheatsheet() -> str:
    return "floyd_warshall({}) -> Floyd-Warshall all-pairs shortest paths. 'I have spoken.' --"
