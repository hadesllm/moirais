# morie.fn — function file (hadesllm/morie)
"""PC algorithm for constraint-based causal discovery.

The PC algorithm recovers the Markov equivalence class of a DAG by
iterating over conditional independence tests and orienting edges
via Meek rules.

References
----------
Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation,
Prediction, and Search* (2nd ed.). MIT Press.

Colombo, D., & Maathuis, M. H. (2014). Order-independent constraint-
based causal structure learning. *Journal of Machine Learning
Research*, 15, 3741-3782.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats

__all__ = ["pcalg"]


def pcalg(
    X: np.ndarray,
    *,
    alpha: float = 0.05,
    max_cond_set: int = 3,
) -> dict[str, Any]:
    r"""Run the PC algorithm to learn a CPDAG from observational data.

    Assumes multivariate Gaussian data and uses Fisher's z-test for
    conditional independence.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.  Columns are variables.
    alpha : float
        Significance level for conditional independence tests.
    max_cond_set : int
        Maximum conditioning set size (limits computation).

    Returns
    -------
    dict
        ``adj`` (adjacency matrix, undirected skeleton),
        ``cpdag`` (partially directed adjacency matrix, 1 = i→j),
        ``sepsets`` (conditioning sets used),
        ``n_tests``, ``p``, ``n``, ``method``.

    Notes
    -----
    The adjacency matrix uses convention: ``adj[i, j] = 1`` means
    edge between i and j; ``cpdag[i, j] = 1, cpdag[j, i] = 0`` means
    directed edge i → j.

    References
    ----------
    Spirtes et al. (2000). MIT Press.
    Colombo & Maathuis (2014). JMLR, 15, 3741-3782.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array (n, p).")
    n, p = X.shape
    if n < p + 2:
        raise ValueError("Need at least p+2 observations for skeleton discovery.")

    # Precompute correlation matrix
    C = np.corrcoef(X.T)
    np.fill_diagonal(C, 1.0)

    # Phase 1: skeleton (undirected complete graph → remove edges)
    adj = np.ones((p, p), dtype=int)
    np.fill_diagonal(adj, 0)
    sepsets: dict[tuple[int, int], list[int]] = {}
    n_tests = 0

    for cond_size in range(max_cond_set + 1):
        pairs = [(i, j) for i in range(p) for j in range(i + 1, p) if adj[i, j]]
        for i, j in pairs:
            neighbors_i = [k for k in range(p) if adj[i, k] and k != j]
            if len(neighbors_i) < cond_size:
                continue
            for S in combinations(neighbors_i, cond_size):
                S_list = list(S)
                pval = _fisher_z(C, i, j, S_list, n)
                n_tests += 1
                if pval > alpha:
                    adj[i, j] = adj[j, i] = 0
                    sepsets[(i, j)] = sepsets[(j, i)] = S_list
                    break

    # Phase 2: orient v-structures (colliders)
    cpdag = adj.copy()
    for i in range(p):
        for j in range(i + 1, p):
            if adj[i, j] == 0:
                continue
            for k in range(p):
                if k == i or k == j:
                    continue
                # i - k - j (k not in sepset(i,j)) → i → k ← j
                if adj[i, k] and adj[j, k] and not adj[i, j]:
                    continue
                if adj[i, k] and adj[j, k]:
                    sep = sepsets.get((i, j), [])
                    if k not in sep:
                        # Orient i → k ← j
                        cpdag[i, k] = 1
                        cpdag[k, i] = 0
                        cpdag[j, k] = 1
                        cpdag[k, j] = 0

    return {
        "adj": adj,
        "cpdag": cpdag,
        "sepsets": sepsets,
        "n_tests": n_tests,
        "p": p,
        "n": n,
        "method": "PC",
    }


def _fisher_z(C: np.ndarray, i: int, j: int, S: list[int], n: int) -> float:
    """Fisher's z-test for partial correlation between i and j given S."""
    if len(S) == 0:
        r = np.clip(C[i, j], -1 + 1e-8, 1 - 1e-8)
    else:
        # Partial correlation via precision matrix subset
        idx = [i, j] + S
        C_sub = C[np.ix_(idx, idx)]
        try:
            prec = np.linalg.inv(C_sub)
        except np.linalg.LinAlgError:
            return 1.0
        r = np.clip(-prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1]), -1 + 1e-8, 1 - 1e-8)

    z = 0.5 * np.log((1.0 + r) / (1.0 - r))
    se = 1.0 / np.sqrt(n - len(S) - 3)
    if se <= 0:
        return 1.0
    return float(2.0 * stats.norm.sf(abs(z / se)))


def cheatsheet() -> str:
    return "pcalg(X) -> PC algorithm CPDAG (Spirtes et al. 2000)."
