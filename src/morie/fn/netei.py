# morie.fn -- function file (hadesllm/morie)
"""Expected influence (signed node strength) for a network."""

from __future__ import annotations

import numpy as np


def network_expected_influence(
    adj_matrix: np.ndarray,
    *,
    node_names: list[str] | None = None,
    step: int = 1,
) -> dict:
    """Expected influence centrality (Robinaugh et al., 2016).

    One-step expected influence is the sum of edge weights (signed)
    incident on a node.  Two-step EI sums indirect paths of length 2.

    Parameters
    ----------
    adj_matrix : ndarray
        Weighted adjacency matrix (p x p), may contain negative edges.
    node_names : list, optional
        Node labels.
    step : int
        1 for one-step EI (default), 2 for two-step.

    Returns
    -------
    dict
        Keys: ``expected_influence`` (dict), ``mean``, ``sd``, ``step``.

    References
    ----------
    Robinaugh, D. J., Millner, A. J., & McNally, R. J. (2016).
    Identifying highly influential nodes in the complicated grief
    network. *Journal of Abnormal Psychology*, 125(6), 747--757.
    """
    A = np.asarray(adj_matrix, dtype=np.float64)
    p = A.shape[0]
    names = node_names or [f"n{i}" for i in range(p)]
    np.fill_diagonal(A, 0.0)

    if step == 1:
        ei = np.sum(A, axis=1)
    else:
        # Two-step: sum of A + A^2 row sums (indirect influence)
        A2 = A @ A
        np.fill_diagonal(A2, 0.0)
        ei = np.sum(A, axis=1) + np.sum(A2, axis=1)

    result = {names[i]: float(ei[i]) for i in range(p)}
    return {
        "expected_influence": result,
        "mean": float(np.mean(ei)),
        "sd": float(np.std(ei, ddof=1)) if p > 1 else 0.0,
        "step": step,
    }


def cheatsheet() -> str:
    return "network_expected_influence({}) -> Expected influence (signed node strength) for a network."
