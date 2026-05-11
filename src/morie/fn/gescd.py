# morie.fn — function file (hadesllm/morie)
"""GES (Greedy Equivalence Search) for score-based causal discovery.

GES is a two-phase greedy algorithm that maximizes a decomposable
score (BIC/BDeu) over the space of CPDAGs.

References
----------
Chickering, D. M. (2002). Optimal structure identification with
greedy search. *Journal of Machine Learning Research*, 3, 507-554.

Hauser, A., & Buhlmann, P. (2012). Characterization and greedy
learning of interventional Markov equivalence classes of directed
acyclic graphs. *JMLR*, 13, 2409-2464.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["gescd"]

_LOG2PI = float(np.log(2.0 * np.pi))


def gescd(
    X: np.ndarray,
    *,
    penalty: float = 1.0,
) -> dict[str, Any]:
    r"""Score-based causal discovery via Greedy Equivalence Search.

    Maximises the Gaussian BIC score:

    .. math::

        \text{BIC}(\mathcal{G}) = \sum_{j=1}^{p}
        \left[
            -\frac{n}{2}\ln\hat{\sigma}^2_j
            - \frac{\text{pen}}{2}|\text{pa}(j)|\ln n
        \right]

    in two phases: (1) forward greedy edge additions, (2) backward
    greedy edge deletions.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.
    penalty : float
        BIC penalty multiplier (default 1.0 = standard BIC).

    Returns
    -------
    dict
        ``dag`` (adjacency matrix, ``dag[i,j]=1`` means i→j),
        ``score``, ``p``, ``n``, ``n_edges``, ``method``.

    References
    ----------
    Chickering (2002). JMLR, 3, 507-554.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n, p).")
    n, p = X.shape

    dag = np.zeros((p, p), dtype=int)

    def _local_score(j: int, parents: list[int]) -> float:
        """Gaussian BIC local score for node j given parents."""
        if len(parents) == 0:
            resid_var = float(np.var(X[:, j], ddof=0))
        else:
            Xp = np.column_stack([np.ones(n), X[:, parents]])
            beta = np.linalg.lstsq(Xp, X[:, j], rcond=None)[0]
            resid = X[:, j] - Xp @ beta
            resid_var = float(np.dot(resid, resid) / n)
        resid_var = max(resid_var, 1e-12)
        ll = -n / 2.0 * (np.log(resid_var) + _LOG2PI + 1.0)
        k = len(parents) + 1              # parents + noise variance
        return ll - penalty / 2.0 * k * np.log(n)

    def _total_score() -> float:
        s = 0.0
        for j in range(p):
            pa = [i for i in range(p) if dag[i, j]]
            s += _local_score(j, pa)
        return s

    # Forward phase: greedily add edges that maximise score gain
    improved = True
    while improved:
        improved = False
        best_gain = 1e-10
        best_edge = None
        for i in range(p):
            for j in range(p):
                if i == j or dag[i, j] or dag[j, i]:
                    continue
                # Check acyclicity after adding i→j
                if _creates_cycle(dag, i, j):
                    continue
                pa_j = [k for k in range(p) if dag[k, j]]
                gain = _local_score(j, pa_j + [i]) - _local_score(j, pa_j)
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (i, j)
        if best_edge is not None:
            dag[best_edge[0], best_edge[1]] = 1
            improved = True

    # Backward phase: greedily remove edges if score improves
    improved = True
    while improved:
        improved = False
        best_gain = 1e-10
        best_edge = None
        for i in range(p):
            for j in range(p):
                if dag[i, j] == 0:
                    continue
                pa_j = [k for k in range(p) if dag[k, j]]
                gain = _local_score(j, [k for k in pa_j if k != i]) - _local_score(j, pa_j)
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (i, j)
        if best_edge is not None:
            dag[best_edge[0], best_edge[1]] = 0
            improved = True

    score = _total_score()
    return {
        "dag": dag,
        "score": score,
        "p": p,
        "n": n,
        "n_edges": int(dag.sum()),
        "method": "GES",
    }


def _creates_cycle(dag: np.ndarray, i: int, j: int) -> bool:
    """DFS check: would adding edge i→j create a cycle?"""
    # Can we reach i from j in the current DAG?
    visited = set()
    stack = [j]
    while stack:
        node = stack.pop()
        if node == i:
            return True
        if node in visited:
            continue
        visited.add(node)
        children = np.where(dag[node] == 1)[0]
        stack.extend(children.tolist())
    return False


def cheatsheet() -> str:
    return "gescd(X) -> GES score-based causal discovery (Chickering 2002, JMLR)."
