# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""System of equations solver (Gauss-Seidel). 'I have been waiting for you.' -- The Architect"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gauss_seidel(
    A: np.ndarray,
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> DescriptiveResult:
    """Solve Ax = b via Gauss-Seidel iterative method.

    Requires A to be diagonally dominant or symmetric positive definite
    for guaranteed convergence.

    Parameters
    ----------
    A : ndarray
        Square coefficient matrix (n x n).
    b : ndarray
        Right-hand-side vector (n,).
    x0 : ndarray or None
        Initial guess (defaults to zeros).
    tol : float
        Convergence tolerance on the L2 norm of the residual.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    DescriptiveResult
        ``value`` is the solution vector as a list; ``extra`` has
        iterations used, residual norm, and convergence status.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")
    if len(b) != n:
        raise ValueError("b must have length n")
    if np.any(np.diag(A) == 0):
        raise ValueError("Diagonal of A must be nonzero")

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64).ravel()
    converged = False
    it = 0
    residual = float("inf")

    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1 :], x_old[i + 1 :])
            x[i] = (b[i] - sigma) / A[i, i]
        residual = float(np.linalg.norm(A @ x - b))
        if residual < tol:
            converged = True
            break

    return DescriptiveResult(
        name="Gauss-Seidel Solver",
        value=x.tolist(),
        extra={
            "iterations": it,
            "residual_norm": residual,
            "converged": converged,
            "tol": tol,
            "n": n,
        },
    )


arcit = gauss_seidel


def cheatsheet() -> str:
    return "gauss_seidel({}) -> System of equations solver (Gauss-Seidel). 'I have been wait"
