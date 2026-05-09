# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""BiCGSTAB solver for non-symmetric linear systems."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bicgstab(
    A: np.ndarray,
    b: np.ndarray,
    *,
    tol: float = 1e-8,
    maxiter: int = 500,
) -> DescriptiveResult:
    """BiCGSTAB (Bi-Conjugate Gradient Stabilized) iterative solver.

    Solves Ax = b for general (non-symmetric) A.

    Parameters
    ----------
    A : ndarray
        Square matrix (n x n).
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
    r = b.copy()
    r0 = r.copy()
    rho = alpha = omega = 1.0
    v = p = np.zeros(n)
    bnorm = np.linalg.norm(b)
    if bnorm < 1e-15:
        return DescriptiveResult(name="BiCGSTAB", value=0.0, extra={"x": x, "iterations": 0})
    for it in range(1, maxiter + 1):
        rho_new = r0 @ r
        if abs(rho_new) < 1e-30:
            break
        beta = (rho_new / (rho + 1e-30)) * (alpha / (omega + 1e-30))
        p = r + beta * (p - omega * v)
        v = A @ p
        alpha = rho_new / (r0 @ v + 1e-30)
        s = r - alpha * v
        if np.linalg.norm(s) / bnorm < tol:
            x += alpha * p
            return DescriptiveResult(
                name="BiCGSTAB",
                value=float(np.linalg.norm(s) / bnorm),
                extra={"x": x, "iterations": it},
            )
        t = A @ s
        omega = (t @ s) / (t @ t + 1e-30)
        x += alpha * p + omega * s
        r = s - omega * t
        rho = rho_new
        rel_res = np.linalg.norm(r) / bnorm
        if rel_res < tol:
            return DescriptiveResult(
                name="BiCGSTAB", value=float(rel_res), extra={"x": x, "iterations": it}
            )
    return DescriptiveResult(
        name="BiCGSTAB",
        value=float(np.linalg.norm(b - A @ x) / bnorm),
        extra={"x": x, "iterations": maxiter},
    )


biccg = bicgstab
