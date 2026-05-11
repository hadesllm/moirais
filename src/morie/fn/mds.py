# morie.fn — function file (hadesllm/morie)
"""Classical Multidimensional Scaling (Torgerson scaling)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from morie.fn._containers import MdsRes


def mds(
    data: pd.DataFrame | np.ndarray,
    n_dims: int = 2,
    is_distance: bool = False,
) -> MdsRes:
    r"""Classical (metric) Multidimensional Scaling.

    If *data* is a feature matrix, Euclidean distances are computed first.
    The distance matrix is double-centred and eigendecomposed to yield
    coordinates in ``n_dims`` dimensions.

    Stress-1 (Kruskal, 1964) is computed as:

    .. math::

        \sigma_1 = \sqrt{\frac{\sum (d_{ij} - \hat{d}_{ij})^2}{\sum d_{ij}^2}}

    Parameters
    ----------
    data : DataFrame, ndarray
        Feature matrix (n x p) or pre-computed distance matrix (n x n).
    n_dims : int
        Number of embedding dimensions.
    is_distance : bool
        If *True*, treat *data* as a square distance matrix.

    Returns
    -------
    MdsRes
        ``coordinates`` (n x n_dims), ``stress``, ``eigenvalues``.

    References
    ----------
    Torgerson, W. S. (1952). Multidimensional scaling: I. Theory and method.
    *Psychometrika*, 17(4), 401-419.  DOI: 10.1007/BF02288916
    """
    X = np.asarray(data, dtype=np.float64)

    if is_distance:
        D = X
    else:
        D = squareform(pdist(X, metric="euclidean"))

    n = D.shape[0]

    # Double-centre
    D2 = D**2
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    grand_mean = D2.mean()
    B = -0.5 * (D2 - row_mean - col_mean + grand_mean)

    # Eigendecompose (symmetric)
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Retain n_dims
    k = min(n_dims, n)
    eigvals_k = np.maximum(eigvals[:k], 0.0)
    coords = eigvecs[:, :k] * np.sqrt(eigvals_k)

    # Stress-1
    D_hat = squareform(pdist(coords, metric="euclidean"))
    ss_dist = np.sum(D**2)
    ss_diff = np.sum((D - D_hat) ** 2)
    stress = float(np.sqrt(ss_diff / ss_dist)) if ss_dist > 0 else 0.0

    return MdsRes(
        coordinates=coords,
        stress=stress,
        eigenvalues=eigvals,
    )


def cheatsheet() -> str:
    return "mds({}) -> Classical Multidimensional Scaling (Torgerson scaling)."
