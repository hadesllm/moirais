# moirais.fn — function file (hadesllm/moirais)
"""PageRank centrality for directed graphs."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def pagerank(
    adj_matrix: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> DescriptiveResult:
    """It does not matter how slowly you go as long as you do not stop. — Confucius"""
    A = np.asarray(adj_matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adj_matrix must be square.")
    if not 0 < damping < 1:
        raise ValueError(f"damping must be in (0, 1), got {damping}.")

    n = A.shape[0]
    out_degree = A.sum(axis=1)
    dangling = out_degree == 0

    M = np.zeros_like(A)
    for i in range(n):
        if out_degree[i] > 0:
            M[i, :] = A[i, :] / out_degree[i]

    r = np.ones(n) / n
    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        r_new = damping * (M.T @ r) + damping * (dangling @ r) / n + (1 - damping) / n
        if np.sum(np.abs(r_new - r)) < tol:
            converged = True
            r = r_new
            break
        r = r_new

    r = r / r.sum()

    return DescriptiveResult(
        name="PageRank",
        value=r,
        extra={"n_iter": n_iter, "converged": converged, "damping": damping},
    )


pgrnk = pagerank


def cheatsheet() -> str:
    return "pagerank({}) -> PageRank centrality for directed graphs."
