# moirais.fn — function file (hadesllm/moirais)
"""Partial correlation network."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import DescriptiveResult


def network_correlation(
    correlation_matrix: np.ndarray,
    *,
    threshold: float = 0.0,
    item_names: list[str] | None = None,
) -> DescriptiveResult:
    """Compute partial correlation network from a correlation matrix.

    Inverts the correlation matrix to get the precision matrix,
    then normalises to partial correlations.

    Parameters
    ----------
    correlation_matrix : ndarray
        Correlation matrix (p x p).
    threshold : float
        Edges with |partial_r| below this are set to 0 (default 0).
    item_names : list[str], optional

    Returns
    -------
    DescriptiveResult
        value=dict with adjacency matrix and edge list.

    References
    ----------
    Epskamp, S., Borsboom, D., & Fried, E. I. (2018). Estimating
    psychological networks and their accuracy. Behavior Research
    Methods, 50(1), 195-212.
    """
    R = np.asarray(correlation_matrix, dtype=np.float64)
    p = R.shape[0]

    if item_names is None:
        item_names = [f"node_{i}" for i in range(p)]

    try:
        P = np.linalg.inv(R + np.eye(p) * 1e-6)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(R)

    d = np.sqrt(np.diag(P))
    d[d < 1e-10] = 1.0
    partial = -P / np.outer(d, d)
    np.fill_diagonal(partial, 0.0)

    if threshold > 0:
        partial[np.abs(partial) < threshold] = 0.0

    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if partial[i, j] != 0:
                edges.append(
                    {
                        "node_a": item_names[i],
                        "node_b": item_names[j],
                        "weight": float(partial[i, j]),
                    }
                )

    return DescriptiveResult(
        name="Partial correlation network",
        value={"adjacency": partial.tolist(), "edges": edges},
        extra={"n_nodes": p, "n_edges": len(edges), "threshold": threshold},
    )


pcor_net = network_correlation


def cheatsheet() -> str:
    return "network_correlation({}) -> Partial correlation network."
