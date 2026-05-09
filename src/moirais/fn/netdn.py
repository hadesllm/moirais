# moirais.fn — function file (hadesllm/moirais)
"""Network density (proportion of non-zero edges)."""

from __future__ import annotations

import numpy as np


def network_density(
    adj_matrix: np.ndarray,
) -> float:
    """Network density: proportion of possible edges that are non-zero.

    For an undirected network of p nodes, maximum edges = p*(p-1)/2.

    Parameters
    ----------
    adj_matrix : ndarray
        Weighted adjacency matrix (p x p).

    Returns
    -------
    float
        Density in [0, 1].

    References
    ----------
    Costantini, G., et al. (2015). State of the aRt personality research.
    *European Journal of Psychological Assessment*, 31(4), 236--249.
    """
    A = np.asarray(adj_matrix, dtype=np.float64)
    p = A.shape[0]
    if p < 2:
        return 0.0
    np.fill_diagonal(A, 0.0)
    # Upper triangle only (undirected)
    upper = A[np.triu_indices(p, k=1)]
    n_edges = np.sum(upper != 0)
    max_edges = p * (p - 1) / 2
    return float(n_edges / max_edges)


def cheatsheet() -> str:
    return "network_density({}) -> Network density (proportion of non-zero edges)."
