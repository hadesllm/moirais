# moirais.fn — function file (hadesllm/moirais)
"""Conjugate gradient solver for symmetric positive-definite systems."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def conjugate_gradient(
    A: np.ndarray,
    b: np.ndarray,
    *,
    tol: float = 1e-8,
    maxiter: int = 500,
) -> DescriptiveResult:
    """Conjugate gradient method for Ax = b (A must be SPD).

    Parameters
    ----------
    A : ndarray
        Symmetric positive-definite matrix (n x n).
    b : ndarray
        Right-hand side vector of length n.
    tol : float
        Convergence tolerance on relative residual norm.
    maxiter : int
        Maximum number of iterations.

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
    r = b.copy()
    p = r.copy()
    rs_old = r @ r
    bnorm = np.linalg.norm(b)
    if bnorm < 1e-15:
        return DescriptiveResult(name="CG", value=0.0, extra={"x": x, "iterations": 0})
    for it in range(1, maxiter + 1):
        Ap = A @ p
        alpha = rs_old / (p @ Ap + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        rel_res = np.sqrt(rs_new) / bnorm
        if rel_res < tol:
            return DescriptiveResult(
                name="CG", value=float(rel_res), extra={"x": x, "iterations": it}
            )
        p = r + (rs_new / (rs_old + 1e-30)) * p
        rs_old = rs_new
    return DescriptiveResult(
        name="CG", value=float(np.linalg.norm(r) / bnorm), extra={"x": x, "iterations": maxiter}
    )


cgsol = conjugate_gradient
