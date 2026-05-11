# morie.fn — function file (hadesllm/morie)
"""Reconstruct distances from MDS coordinates. 'ZA WARUDO!' -- DIO, JoJo's Bizarre Adventure"""

from __future__ import annotations

from ._containers import DescriptiveResult


def mds_reconstruct_distances(X):
    """Compute pairwise Euclidean distance matrix from coordinate matrix X.

    Parameters
    ----------
    X : array-like
        Coordinate matrix (n x p).

    Returns
    -------
    DescriptiveResult
        value = reconstructed distance matrix (n x n).
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            D[i, j] = d
            D[j, i] = d
    return DescriptiveResult(name="mds_reconstruct_distances", value=D, extra={"n": n})


mdsrk = mds_reconstruct_distances


def cheatsheet() -> str:
    return "mds_reconstruct_distances({}) -> Reconstruct distances from MDS coordinates. 'ZA WARUDO!' -- "
