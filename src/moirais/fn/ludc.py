# moirais.fn — function file (hadesllm/moirais)
"""LU decomposition with partial pivoting."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lu_decomposition(
    A: np.ndarray,
) -> DescriptiveResult:
    """LU decomposition with partial pivoting.

    Factors *A* into P, L, U such that PA = LU where P is a permutation
    matrix, L is unit lower triangular, and U is upper triangular.

    Parameters
    ----------
    A : ndarray
        Square matrix (n x n).

    Returns
    -------
    DescriptiveResult
        ``value`` is the determinant; ``extra`` has P, L, U arrays.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    swaps = 0
    for k in range(n - 1):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            P[[k, pivot]] = P[[pivot, k]]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
            swaps += 1
        for i in range(k + 1, n):
            if abs(U[k, k]) < 1e-15:
                continue
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    det_val = float((-1) ** swaps * np.prod(np.diag(U)))
    return DescriptiveResult(
        name="LU Decomposition",
        value=det_val,
        extra={"P": P, "L": L, "U": U},
    )


ludc = lu_decomposition
