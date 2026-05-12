# morie.fn -- function file (hadesllm/morie)
"""K-nearest neighbors spatial weights matrix."""

import numpy as np
from scipy.spatial.distance import cdist

from ._containers import DescriptiveResult


def knn_weights(
    coords: np.ndarray,
    *,
    k: int = 4,
) -> DescriptiveResult:
    """
    Construct a k-nearest neighbors spatial weights matrix.

    For each observation, the *k* closest observations (by Euclidean distance)
    receive weight 1; all others receive weight 0. The diagonal is always 0
    (a point is not its own neighbor).

    :param coords: Observation coordinates, shape (n, 2).
    :param k: Number of nearest neighbors. Default 4.
    :return: :class:`DescriptiveResult` with value = binary weight matrix (n, n).
    :raises ValueError: If k >= n or k < 1.

    References
    ----------
    Getis, A. & Aldstadt, J. (2004). Constructing the spatial weights matrix
    using a local statistic. *Geographical Analysis*, 36(2), 90-104.
    https://doi.org/10.1111/j.1538-4632.2004.tb01127.x
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if k >= n:
        raise ValueError(f"k ({k}) must be < n ({n}).")

    dists = cdist(coords, coords)
    np.fill_diagonal(dists, np.inf)

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        idx = np.argpartition(dists[i], k)[:k]
        W[i, idx] = 1.0

    return DescriptiveResult(
        name="KNN Weights",
        value=W,
        extra={"k": k, "n": n},
    )


knear_fn = knn_weights


def cheatsheet() -> str:
    return "knn_weights({}) -> K-nearest neighbors spatial weights matrix."
