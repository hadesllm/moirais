# morie.fn -- function file (hadesllm/morie)
"""Node strength centrality for a network."""

from __future__ import annotations

import numpy as np


def network_strength(
    adj_matrix: np.ndarray,
    *,
    node_names: list[str] | None = None,
) -> dict:
    """Node strength centrality (sum of absolute edge weights).

    Parameters
    ----------
    adj_matrix : ndarray
        Weighted adjacency matrix (p x p).
    node_names : list, optional
        Labels for nodes.  Defaults to ``n0, n1, ...``.

    Returns
    -------
    dict
        Keys: ``strength`` (dict node->value), ``mean``, ``sd``.

    References
    ----------
    Barrat, A., et al. (2004). The architecture of complex weighted
    networks. *PNAS*, 101(11), 3747--3752.
    """
    A = np.asarray(adj_matrix, dtype=np.float64)
    p = A.shape[0]
    names = node_names or [f"n{i}" for i in range(p)]

    np.fill_diagonal(A, 0.0)
    s = np.sum(np.abs(A), axis=1)
    strength = {names[i]: float(s[i]) for i in range(p)}

    return {
        "strength": strength,
        "mean": float(np.mean(s)),
        "sd": float(np.std(s, ddof=1)) if p > 1 else 0.0,
    }


def cheatsheet() -> str:
    return "network_strength({}) -> Node strength centrality for a network."
