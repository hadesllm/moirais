# morie.fn -- function file (hadesllm/morie)
"""He who would learn to fly one day must first learn to stand and walk. -- Friedrich Nietzsche"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ldlt_factorize(
    A: np.ndarray,
) -> DescriptiveResult:
    """
    LDL^T decomposition for symmetric matrices.

    Decomposes A = L D L^T where L is unit lower triangular and D is
    diagonal. No square roots required (unlike Cholesky).

    :param A: Symmetric matrix.
    :return: DescriptiveResult with L and D matrices.
    :raises ValueError: If A is not square or not symmetric.

    References
    ----------
    Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*.
    4th ed. Johns Hopkins University Press. Sec. 4.1.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("A must be symmetric.")

    n = A.shape[0]
    L = np.eye(n)
    D = np.zeros(n)
    V = np.zeros(n)

    for j in range(n):
        for i in range(j):
            V[i] = L[j, i] * D[i]
        D[j] = A[j, j] - np.dot(L[j, :j], V[:j])

        if np.abs(D[j]) < 1e-15:
            raise ValueError(f"Zero pivot at position {j}; matrix may be singular.")

        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.dot(L[i, :j], V[:j])) / D[j]

    D_mat = np.diag(D)
    reconstruction_error = float(np.max(np.abs(A - L @ D_mat @ L.T)))

    return DescriptiveResult(
        name="LDL^T Factorization",
        value=float(np.prod(D)),
        extra={
            "L": L,
            "D": D,
            "D_matrix": D_mat,
            "determinant": float(np.prod(D)),
            "positive_definite": bool(np.all(D > 0)),
            "reconstruction_error": reconstruction_error,
            "n": n,
        },
    )


short = ldlt_factorize


def cheatsheet() -> str:
    return "ldlt_factorize({}) -> LDL^T factorization for symmetric matrices. 'The circle is n"
