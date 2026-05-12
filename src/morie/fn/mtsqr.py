# morie.fn -- function file (hadesllm/morie)
"""Matrix square root."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def matrix_sqrt(
    A: np.ndarray,
    *,
    maxiter: int = 100,
    tol: float = 1e-12,
) -> DescriptiveResult:
    """Matrix square root via Denman-Beavers iteration.

    Computes S such that S @ S ~ A.

    Parameters
    ----------
    A : ndarray
        Square matrix (n x n), should have no negative real eigenvalues.
    maxiter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    DescriptiveResult
        ``value`` is the residual ||S^2 - A||_F; ``extra`` has the
        square-root matrix.
    """
    A = np.asarray(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    n = A.shape[0]
    Y = A.copy()
    Z = np.eye(n)
    for it in range(maxiter):
        Y_inv = np.linalg.inv(Z)
        Z_inv = np.linalg.inv(Y)
        Y_new = 0.5 * (Y + Y_inv)
        Z_new = 0.5 * (Z + Z_inv)
        if np.linalg.norm(Y_new - Y, "fro") < tol:
            Y = Y_new
            break
        Y, Z = Y_new, Z_new
    residual = float(np.linalg.norm(Y @ Y - A, "fro"))
    return DescriptiveResult(
        name="Matrix Square Root",
        value=residual,
        extra={"matrix": Y, "iterations": it + 1},
    )


mtsqr = matrix_sqrt
