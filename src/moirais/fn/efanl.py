# moirais.fn — function file (hadesllm/moirais)
"""Exploratory factor analysis (principal axis factoring)."""

from __future__ import annotations

import numpy as np

from ._containers import FaRes


def efa_principal_axis(
    data: np.ndarray,
    n_factors: int = 2,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> FaRes:
    """Exploratory factor analysis via iterated principal axis factoring.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix (centred internally).
    n_factors : int
        Number of factors to extract.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on communality change.

    Returns
    -------
    FaRes
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape
    X = X - X.mean(axis=0)
    R = np.corrcoef(X, rowvar=False)

    communalities = 1.0 - 1.0 / np.diag(np.linalg.inv(R + np.eye(p) * 1e-8))
    communalities = np.clip(communalities, 0.01, 0.99)

    for _ in range(max_iter):
        R_reduced = R.copy()
        np.fill_diagonal(R_reduced, communalities)

        eigvals, eigvecs = np.linalg.eigh(R_reduced)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        pos_eigvals = np.maximum(eigvals[:n_factors], 0)
        loadings = eigvecs[:, :n_factors] * np.sqrt(pos_eigvals)

        new_comm = np.sum(loadings ** 2, axis=1)
        new_comm = np.clip(new_comm, 0.01, 0.99)

        if np.max(np.abs(new_comm - communalities)) < tol:
            communalities = new_comm
            break
        communalities = new_comm

    var_explained = np.sum(loadings ** 2, axis=0)

    return FaRes(
        loadings=loadings,
        communalities=communalities,
        eigenvalues=eigvals[:n_factors],
        variance_explained=var_explained,
        n_factors=n_factors,
        rotation="none",
    )


efanl = efa_principal_axis


def cheatsheet() -> str:
    return "efa_principal_axis({}) -> Exploratory factor analysis (principal axis)."
