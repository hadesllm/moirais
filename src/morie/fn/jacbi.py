# morie.fn — function file (hadesllm/morie)
"""Jacobi iterative solver for diagonally dominant systems."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def jacobi_solve(
    A: np.ndarray,
    b: np.ndarray,
    *,
    tol: float = 1e-8,
    maxiter: int = 1000,
) -> DescriptiveResult:
    """Jacobi iterative method for Ax = b.

    Converges for strictly diagonally dominant matrices.

    Parameters
    ----------
    A : ndarray
        Square matrix (n x n), should be diagonally dominant.
    b : ndarray
        Right-hand side vector.
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    DescriptiveResult
        ``value`` is the final relative residual; ``extra`` has x and
        iteration count.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = int(b) if b.ndim == 0 else len(b)
    D = np.diag(A)
    if np.any(np.abs(D) < 1e-15):
        raise ValueError("Zero diagonal element — Jacobi cannot proceed")
    R = A - np.diag(D)
    x = np.zeros(n)
    bnorm = np.linalg.norm(b)
    if bnorm < 1e-15:
        return DescriptiveResult(name="Jacobi", value=0.0, extra={"x": x, "iterations": 0})
    for it in range(1, maxiter + 1):
        x_new = (b - R @ x) / D
        rel_res = np.linalg.norm(x_new - x) / (np.linalg.norm(x_new) + 1e-15)
        x = x_new
        if rel_res < tol:
            return DescriptiveResult(
                name="Jacobi", value=float(rel_res), extra={"x": x, "iterations": it}
            )
    return DescriptiveResult(
        name="Jacobi",
        value=float(np.linalg.norm(b - A @ x) / bnorm),
        extra={"x": x, "iterations": maxiter},
    )


jacbi = jacobi_solve
