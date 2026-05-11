"""Successive Over-Relaxation (SOR) solver."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def sor_solve(
    A: np.ndarray,
    b: np.ndarray,
    *,
    omega: float = 1.5,
    tol: float = 1e-8,
    maxiter: int = 1000,
) -> DescriptiveResult:
    """Successive Over-Relaxation iterative solver for Ax = b.

    Generalises Gauss-Seidel with relaxation factor *omega*.
    omega = 1 reduces to Gauss-Seidel; 1 < omega < 2 accelerates convergence
    for SPD matrices.

    Parameters
    ----------
    A : ndarray
        Square matrix (n x n).
    b : ndarray
        Right-hand side vector.
    omega : float
        Relaxation factor (0 < omega < 2 for convergence).
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    DescriptiveResult
        ``value`` is the final relative residual; ``extra`` has x, omega,
        and iteration count.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = int(b) if b.ndim == 0 else len(b)
    x = np.zeros(n)
    bnorm = np.linalg.norm(b)
    if bnorm < 1e-15:
        return DescriptiveResult(
            name="SOR", value=0.0, extra={"x": x, "iterations": 0, "omega": omega}
        )
    for it in range(1, maxiter + 1):
        x_old = x.copy()
        for i in range(n):
            if abs(A[i, i]) < 1e-15:
                raise ValueError(f"Zero diagonal at row {i}")
            gs = (b[i] - A[i, :i] @ x[:i] - A[i, i + 1 :] @ x_old[i + 1 :]) / A[i, i]
            x[i] = (1 - omega) * x_old[i] + omega * gs
        rel_res = np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-15)
        if rel_res < tol:
            return DescriptiveResult(
                name="SOR",
                value=float(rel_res),
                extra={"x": x, "iterations": it, "omega": omega},
            )
    return DescriptiveResult(
        name="SOR",
        value=float(np.linalg.norm(b - A @ x) / bnorm),
        extra={"x": x, "iterations": maxiter, "omega": omega},
    )


sorsl = sor_solve
