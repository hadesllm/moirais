# moirais.fn — function file (hadesllm/moirais)
"""Cholesky decomposition and linear solve."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def cholesky_solve(
    A: np.ndarray,
    b: np.ndarray | None = None,
) -> DescriptiveResult:
    """Cholesky decomposition of a symmetric positive-definite matrix.

    Computes L such that A = L L^T.  If *b* is given, solves Ax = b via
    forward/back substitution.

    Parameters
    ----------
    A : ndarray
        Symmetric positive-definite matrix (n x n).
    b : ndarray, optional
        Right-hand side vector of length n.

    Returns
    -------
    DescriptiveResult
        ``value`` is det(A); ``extra`` has L and optionally x (solution).
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - np.dot(L[i, :j], L[j, :j])
            if i == j:
                if s <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    extra: dict = {"L": L}
    val = float(np.prod(np.diag(L)) ** 2)
    if b is not None:
        b = np.asarray(b, dtype=float)
        y = np.linalg.solve(L, b)
        x = np.linalg.solve(L.T, y)
        extra["x"] = x
    return DescriptiveResult(name="Cholesky", value=val, extra=extra)


chles = cholesky_solve
