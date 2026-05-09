# moirais.fn — function file (hadesllm/moirais)
"""Oblimin rotation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def oblimin(
    loadings: np.ndarray,
    gamma: float = 0.0,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> DescriptiveResult:
    """Direct oblimin (oblique) rotation of factor loadings.

    Parameters
    ----------
    loadings : ndarray (p, k)
        Unrotated factor loading matrix.
    gamma : float
        Oblimin parameter. 0 = direct quartimin.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    DescriptiveResult
        ``value`` is the rotated loadings matrix.
        ``extra`` has ``rotation_matrix``, ``phi`` (factor correlation),
        ``n_iter``.
    """
    A = np.asarray(loadings, dtype=np.float64).copy()
    p, k = A.shape
    T = np.eye(k)
    alpha = 0.01

    N = np.ones((k, k)) - np.eye(k)
    I_gamma = np.eye(p) * gamma / p + np.ones((p, p)) * (1 - gamma) / p if gamma != 0 else None

    for it in range(max_iter):
        B = A @ np.linalg.inv(T).T
        B2 = B ** 2

        if I_gamma is not None:
            grad = A.T @ (B2 @ N - I_gamma @ B @ N)
        else:
            grad = A.T @ (B2 @ N)

        T_new = T - alpha * grad @ np.linalg.inv(T @ T.T) @ T

        U, s, Vt = np.linalg.svd(T_new)
        T_new = U @ Vt

        if np.max(np.abs(T_new - T)) < tol:
            T = T_new
            break
        T = T_new

    T_inv = np.linalg.inv(T)
    rotated = A @ T_inv.T
    phi = T.T @ T

    return DescriptiveResult(
        name="Oblimin",
        value=rotated,
        extra={
            "rotation_matrix": T,
            "phi": phi,
            "n_iter": it + 1,
        },
    )


oblmn = oblimin


def cheatsheet() -> str:
    return "oblimin({}) -> Direct oblimin oblique rotation."
