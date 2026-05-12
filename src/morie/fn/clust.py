# morie.fn -- function file (hadesllm/morie)
"""Graph clustering coefficient."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def clustering_coefficient(adj_matrix: np.ndarray) -> DescriptiveResult:
    r"""Local and global clustering coefficients.

    The local clustering coefficient for node *v* is:

    .. math::

        C_v = \\frac{2 \\, T(v)}{k_v (k_v - 1)}

    where *T(v)* is the number of triangles through *v* and
    :math:`k_v` is its degree.  The global coefficient is the mean
    of all local coefficients.

    Parameters
    ----------
    adj_matrix : ndarray
        Square symmetric adjacency matrix (undirected graph).

    Returns
    -------
    DescriptiveResult
        ``value`` is the global (average) clustering coefficient.
        ``extra`` has ``local`` (per-node array) and ``n_triangles``.

    References
    ----------
    Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of
    'small-world' networks. *Nature*, 393(6684), 440--442.
    """
    A = np.asarray(adj_matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adj_matrix must be square.")

    A_bin = (A != 0).astype(np.float64)
    np.fill_diagonal(A_bin, 0)
    n = A_bin.shape[0]

    A2 = A_bin @ A_bin
    triangles_per_node = np.diag(A_bin @ A2) / 2.0
    degree = A_bin.sum(axis=1)

    local = np.zeros(n)
    for i in range(n):
        k = degree[i]
        if k >= 2:
            local[i] = 2.0 * triangles_per_node[i] / (k * (k - 1))

    global_cc = float(np.mean(local))
    total_triangles = int(triangles_per_node.sum() / 3)

    return DescriptiveResult(
        name="ClusteringCoefficient",
        value=global_cc,
        extra={"local": local, "n_triangles": total_triangles, "n_nodes": n},
    )


clust = clustering_coefficient


def cheatsheet() -> str:
    return "clustering_coefficient({}) -> Graph clustering coefficient."
