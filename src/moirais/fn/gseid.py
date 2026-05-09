# moirais.fn — function file (hadesllm/moirais)
"""Gauss-Seidel iterative solver."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gauss_seidel(
    A: np.ndarray,
    b: np.ndarray,
    *,
    tol: float = 1e-8,
    maxiter: int = 1000,
) -> DescriptiveResult:
    """Gauss-Seidel iterative method for Ax = b.

    Uses the most recently updated values within each iteration, converging
    faster than Jacobi for diagonally dominant systems.

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
    x = np.zeros(n)
    bnorm = np.linalg.norm(b)
    if bnorm < 1e-15:
        return DescriptiveResult(name="Gauss-Seidel", value=0.0, extra={"x": x, "iterations": 0})
    for it in range(1, maxiter + 1):
        x_old = x.copy()
        for i in range(n):
            if abs(A[i, i]) < 1e-15:
                raise ValueError(f"Zero diagonal at row {i}")
            s = b[i] - A[i, :i] @ x[:i] - A[i, i + 1 :] @ x_old[i + 1 :]
            x[i] = s / A[i, i]
        rel_res = np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-15)
        if rel_res < tol:
            return DescriptiveResult(
                name="Gauss-Seidel", value=float(rel_res), extra={"x": x, "iterations": it}
            )
    return DescriptiveResult(
        name="Gauss-Seidel",
        value=float(np.linalg.norm(b - A @ x) / bnorm),
        extra={"x": x, "iterations": maxiter},
    )


gseid = gauss_seidel
