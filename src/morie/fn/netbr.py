# morie.fn — function file (hadesllm/morie)
"""Bridge centrality between communities in a network."""

from __future__ import annotations

import numpy as np


def network_bridge(
    adj_matrix: np.ndarray,
    communities: dict[str, list[int]] | list[int],
    *,
    node_names: list[str] | None = None,
) -> dict:
    """Bridge centrality: strength of connections to other communities.

    Bridge strength of node i = sum of |edge weights| from i to nodes
    in other communities (Jones et al., 2019).

    Parameters
    ----------
    adj_matrix : ndarray
        Weighted adjacency matrix (p x p).
    communities : dict or list
        If dict: mapping of community name to list of node indices.
        If list: community label for each node (length p).
    node_names : list, optional
        Node labels.

    Returns
    -------
    dict
        Keys: ``bridge_strength`` (dict), ``bridge_expected_influence``
        (dict), ``mean_strength``, ``mean_ei``.

    References
    ----------
    Jones, P. J., Ma, R., & McNally, R. J. (2019). Bridge centrality:
    a network approach to understanding comorbidity. *Multivariate
    Behavioral Research*, 56(2), 353--367.
    """
    A = np.asarray(adj_matrix, dtype=np.float64)
    p = A.shape[0]
    names = node_names or [f"n{i}" for i in range(p)]
    np.fill_diagonal(A, 0.0)

    # Normalize communities to node->label mapping
    if isinstance(communities, dict):
        node_comm = np.full(p, -1, dtype=int)
        for idx, (_, members) in enumerate(communities.items()):
            for m in members:
                node_comm[m] = idx
    else:
        node_comm = np.asarray(communities)

    bridge_str = np.zeros(p)
    bridge_ei = np.zeros(p)
    for i in range(p):
        for j in range(p):
            if i != j and node_comm[i] != node_comm[j]:
                bridge_str[i] += abs(A[i, j])
                bridge_ei[i] += A[i, j]

    return {
        "bridge_strength": {names[i]: float(bridge_str[i]) for i in range(p)},
        "bridge_expected_influence": {names[i]: float(bridge_ei[i]) for i in range(p)},
        "mean_strength": float(np.mean(bridge_str)),
        "mean_ei": float(np.mean(bridge_ei)),
    }


def cheatsheet() -> str:
    return "network_bridge({}) -> Bridge centrality between communities in a network."
