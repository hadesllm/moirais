# morie.fn -- function file (hadesllm/morie)
"""Out of chaos, comes order. -- Friedrich Nietzsche"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lu_factorize(
    A: np.ndarray,
) -> DescriptiveResult:
    """
    LU decomposition with partial pivoting: PA = LU.

    :param A: Square matrix.
    :return: DescriptiveResult with L, U, P matrices and determinant.
    :raises ValueError: If A is not square or is singular.

    References
    ----------
    Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*.
    4th ed. Johns Hopkins University Press. Ch. 3.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if np.abs(U[pivot, k]) < 1e-15:
            raise ValueError("Matrix is singular or nearly singular.")

        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            P[[k, pivot]] = P[[pivot, k]]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    det_val = float(np.prod(np.diag(U)))
    n_swaps = int(np.sum(np.argmax(P, axis=1) != np.arange(n)))
    if n_swaps % 2 == 1:
        det_val = -det_val

    return DescriptiveResult(
        name="LU Factorization",
        value=float(np.abs(det_val)),
        extra={
            "L": L,
            "U": U,
            "P": P,
            "determinant": det_val,
            "n": n,
        },
    )


short = lu_factorize


def cheatsheet() -> str:
    return "lu_factorize({}) -> LU factorization with partial pivoting. 'Now, young Skywalke"
