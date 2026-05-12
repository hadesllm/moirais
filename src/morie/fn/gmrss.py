# morie.fn -- function file (hadesllm/morie)
"""GMRES iterative solver for non-symmetric linear systems."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gmres_solve(
    A: np.ndarray,
    b: np.ndarray,
    *,
    tol: float = 1e-8,
    maxiter: int = 200,
) -> DescriptiveResult:
    """GMRES (Generalized Minimal RESidual) iterative solver.

    Solves Ax = b for non-symmetric A using Arnoldi-based Krylov subspace.

    Parameters
    ----------
    A : ndarray
        Square matrix (n x n).
    b : ndarray
        Right-hand side vector of length n.
    tol : float
        Convergence tolerance on relative residual.
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
    r = b - A @ x
    bnorm = np.linalg.norm(b)
    if bnorm < 1e-15:
        return DescriptiveResult(name="GMRES", value=0.0, extra={"x": x, "iterations": 0})
    beta = np.linalg.norm(r)
    m = min(maxiter, n)
    Q = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    Q[:, 0] = r / beta
    for j in range(m):
        v = A @ Q[:, j]
        for i in range(j + 1):
            H[i, j] = Q[:, i] @ v
            v -= H[i, j] * Q[:, i]
        H[j + 1, j] = np.linalg.norm(v)
        if H[j + 1, j] > 1e-14:
            Q[:, j + 1] = v / H[j + 1, j]
        e1 = np.zeros(j + 2)
        e1[0] = beta
        y, _, _, _ = np.linalg.lstsq(H[: j + 2, : j + 1], e1, rcond=None)
        res = np.linalg.norm(H[: j + 2, : j + 1] @ y - e1) / bnorm
        if res < tol:
            x = Q[:, : j + 1] @ y
            return DescriptiveResult(
                name="GMRES", value=float(res), extra={"x": x, "iterations": j + 1}
            )
    y, _, _, _ = np.linalg.lstsq(H[: m + 1, :m], np.append([beta], np.zeros(m)), rcond=None)
    x = Q[:, :m] @ y
    res = float(np.linalg.norm(b - A @ x) / bnorm)
    return DescriptiveResult(name="GMRES", value=res, extra={"x": x, "iterations": m})


gmrss = gmres_solve
