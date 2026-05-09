# moirais.fn — function file (hadesllm/moirais)
"""Network clustering coefficient."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def network_clustering_coeff(adj: np.ndarray) -> ESRes:
    """Average clustering coefficient (Watts & Strogatz, 1998).

    Parameters
    ----------
    adj : (n, n) binary adjacency matrix

    Returns
    -------
    ESRes
    """
    A = (np.asarray(adj, dtype=float) > 0).astype(float)
    np.fill_diagonal(A, 0)
    n = A.shape[0]

    cc = np.zeros(n)
    for i in range(n):
        nbrs = np.where(A[i] > 0)[0]
        ki = len(nbrs)
        if ki < 2:
            cc[i] = 0.0
            continue
        sub = A[np.ix_(nbrs, nbrs)]
        triangles = sub.sum() / 2
        cc[i] = 2 * triangles / (ki * (ki - 1))

    avg_cc = float(np.mean(cc))

    return ESRes(
        measure="clustering_coefficient",
        estimate=avg_cc,
        n=n,
        extra={"per_node": cc.tolist()},
    )


ntccf = network_clustering_coeff


def cheatsheet() -> str:
    return "network_clustering_coeff({}) -> Network clustering coefficient."
