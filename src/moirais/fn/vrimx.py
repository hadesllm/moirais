"""Varimax rotation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def varimax(
    loadings: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> DescriptiveResult:
    """Varimax (orthogonal) rotation of factor loadings.

    Parameters
    ----------
    loadings : ndarray (p, k)
        Unrotated factor loading matrix.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    DescriptiveResult
        ``value`` is the rotated loadings matrix.
        ``extra`` has ``rotation_matrix`` and ``n_iter``.
    """
    A = np.asarray(loadings, dtype=np.float64).copy()
    p, k = A.shape
    R = np.eye(k)

    for it in range(max_iter):
        B = A @ R
        B2 = B ** 2
        cm = B2.mean(axis=0)

        u = B ** 3 - B * cm[None, :]
        svd_mat = A.T @ u
        U, _, Vt = np.linalg.svd(svd_mat)
        R_new = U @ Vt

        if np.max(np.abs(R_new - R)) < tol:
            R = R_new
            break
        R = R_new

    rotated = A @ R
    return DescriptiveResult(
        name="Varimax",
        value=rotated,
        extra={"rotation_matrix": R, "n_iter": it + 1},
    )


vrimx = varimax


def cheatsheet() -> str:
    return "varimax({}) -> Varimax orthogonal rotation of factor loadings."
