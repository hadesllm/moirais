# moirais.fn — function file (hadesllm/moirais)
"""Factor analysis (maximum likelihood)."""

from __future__ import annotations

import numpy as np

from ._containers import FaRes


def factor_analysis_ml(
    data: np.ndarray,
    n_factors: int = 2,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> FaRes:
    """Maximum likelihood factor analysis.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix (centred internally).
    n_factors : int
        Number of factors.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    FaRes
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape
    X = X - X.mean(axis=0)
    S = np.cov(X, rowvar=False, ddof=1)

    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    L = eigvecs[:, :n_factors] * np.sqrt(np.maximum(eigvals[:n_factors] - 1.0, 0.1))
    psi = np.diag(S) - np.sum(L ** 2, axis=1)
    psi = np.maximum(psi, 1e-6)

    for _ in range(max_iter):
        psi_inv = 1.0 / psi
        M = np.eye(n_factors) + (L.T * psi_inv) @ L
        M_inv = np.linalg.inv(M)

        L_new = S @ (L * psi_inv[:, None]) @ np.linalg.inv(
            np.eye(n_factors) + M_inv @ (L.T * psi_inv) @ S @ (L * psi_inv[:, None])
        )

        L_new = S @ np.diag(psi_inv) @ L @ M_inv
        psi_new = np.diag(S) - np.sum(L_new * (S @ np.diag(psi_inv) @ L @ M_inv), axis=1)
        psi_new = np.maximum(psi_new, 1e-6)

        if np.max(np.abs(L_new - L)) < tol:
            L = L_new
            psi = psi_new
            break
        L = L_new
        psi = psi_new

    communalities = np.sum(L ** 2, axis=1)
    var_explained = np.sum(L ** 2, axis=0)

    return FaRes(
        loadings=L,
        communalities=communalities,
        eigenvalues=eigvals[:n_factors],
        variance_explained=var_explained,
        n_factors=n_factors,
        rotation="none",
    )


fanlz = factor_analysis_ml


def cheatsheet() -> str:
    return "factor_analysis_ml({}) -> Maximum likelihood factor analysis."
