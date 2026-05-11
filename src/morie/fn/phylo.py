# morie.fn — function file (hadesllm/morie)
"""Phylogenetic diversity index."""

from __future__ import annotations

from typing import Any

import numpy as np


def phylogenetic_diversity(
    distance_matrix: np.ndarray,
    *,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute phylogenetic diversity (PD) from a distance matrix.

    Uses the minimum spanning tree (MST) total branch length as a
    measure of phylogenetic diversity (Faith's PD approximation from
    a distance matrix).

    .. math::

        PD = \\sum_{e \\in MST} w_e

    Parameters
    ----------
    distance_matrix : array_like
        Symmetric pairwise distance matrix (n x n).
    labels : list of str or None
        Labels for each taxon/sequence.

    Returns
    -------
    dict
        Keys: 'pd' (total PD), 'n_taxa', 'mean_distance',
              'mst_edges' (list of (i, j, weight)).

    References
    ----------
    Faith, D. P. (1992). Conservation evaluation and phylogenetic
    diversity. Biological Conservation, 61(1), 1-10.
    """
    D = np.asarray(distance_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distance_matrix must be square.")
    n = D.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 taxa.")

    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    min_cost = D[0].copy()
    min_from = np.zeros(n, dtype=int)
    edges = []
    total_pd = 0.0

    for _ in range(n - 1):
        candidates = np.where(~in_tree)[0]
        best_idx = candidates[np.argmin(min_cost[candidates])]
        w = min_cost[best_idx]
        edges.append((int(min_from[best_idx]), int(best_idx), float(w)))
        total_pd += w
        in_tree[best_idx] = True

        for j in np.where(~in_tree)[0]:
            if D[best_idx, j] < min_cost[j]:
                min_cost[j] = D[best_idx, j]
                min_from[j] = best_idx

    triu = D[np.triu_indices(n, k=1)]
    mean_dist = float(np.mean(triu)) if len(triu) > 0 else 0.0

    return {
        "pd": float(total_pd),
        "n_taxa": n,
        "mean_distance": mean_dist,
        "mst_edges": edges,
    }


phylo = phylogenetic_diversity


def cheatsheet() -> str:
    return "phylogenetic_diversity({}) -> Faith's PD from distance matrix."
