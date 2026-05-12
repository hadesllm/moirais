# morie.fn -- function file (hadesllm/morie)
"""Community detection via modularity optimization."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def network_community(
    adjacency: np.ndarray,
    *,
    n_communities: int | None = None,
    item_names: list[str] | None = None,
) -> DescriptiveResult:
    """Community detection using spectral modularity optimization.

    Parameters
    ----------
    adjacency : ndarray
        Weighted adjacency matrix (p x p).
    n_communities : int, optional
        Number of communities. Default: determined by eigenvalue gap.
    item_names : list[str], optional

    Returns
    -------
    DescriptiveResult
        value=dict with community assignments and modularity.

    References
    ----------
    Newman, M. E. J. (2006). Modularity and community structure in
    networks. Proceedings of the National Academy of Sciences.
    """
    A = np.abs(np.asarray(adjacency, dtype=np.float64))
    p = A.shape[0]

    if item_names is None:
        item_names = [f"node_{i}" for i in range(p)]

    k = A.sum(axis=1)
    m = A.sum() / 2
    if m < 1e-10:
        labels = np.zeros(p, dtype=int)
        return DescriptiveResult(
            name="Network communities",
            value={"labels": labels.tolist(), "modularity": 0.0},
            extra={"n_nodes": p, "n_communities": 1},
        )

    B = A - np.outer(k, k) / (2 * m)
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(-evals)
    evals = evals[order]
    evecs = evecs[:, order]

    if n_communities is None:
        gaps = np.diff(evals[: min(p, 10)])
        n_communities = max(int(np.argmin(gaps) + 1), 2)
        n_communities = min(n_communities, p)

    features = evecs[:, :n_communities]

    from scipy.cluster.vq import kmeans2

    centroids, labels = kmeans2(features, n_communities, minit="points")

    Q = 0.0
    for i in range(p):
        for j in range(p):
            if labels[i] == labels[j]:
                Q += B[i, j]
    Q /= 2 * m

    assignments = {item_names[i]: int(labels[i]) for i in range(p)}

    return DescriptiveResult(
        name="Network communities",
        value={"labels": labels.tolist(), "modularity": float(Q), "assignments": assignments},
        extra={"n_nodes": p, "n_communities": int(n_communities)},
    )


community = network_community


def cheatsheet() -> str:
    return "network_community({}) -> Community detection via modularity optimization."
