# moirais.fn — function file (hadesllm/moirais)
"""Knowing yourself is the beginning of all wisdom. — Aristotle"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def kronecker_decompose(
    C: np.ndarray,
    *,
    m1: int | None = None,
    n1: int | None = None,
) -> DescriptiveResult:
    """Approximate nearest Kronecker product decomposition.

    Given C approx= A kron B, find A (m1 x n1) and B (m2 x n2)
    that minimize ||C - A kron B||_F using the Van Loan & Pitsianis (1993)
    rearrangement + SVD approach.

    Parameters
    ----------
    C : ndarray of shape (m, n)
        Matrix to decompose.
    m1, n1 : int, optional
        Dimensions of factor A. Must divide m and n respectively.
        Defaults to sqrt-sized factors.

    Returns
    -------
    DescriptiveResult
        With ``value`` = dict(A=A, B=B) and ``extra`` containing
        approximation error.
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2:
        raise ValueError("C must be 2-D")
    m, n = C.shape

    if m1 is None:
        m1 = int(np.round(np.sqrt(m)))
        while m % m1 != 0 and m1 > 1:
            m1 -= 1
    if n1 is None:
        n1 = int(np.round(np.sqrt(n)))
        while n % n1 != 0 and n1 > 1:
            n1 -= 1

    m2 = m // m1
    n2 = n // n1
    if m1 * m2 != m or n1 * n2 != n:
        raise ValueError(f"m1={m1}, n1={n1} do not evenly divide ({m}, {n})")

    R = np.zeros((m1 * n1, m2 * n2))
    for i in range(m1):
        for j in range(n1):
            block = C[i * m2 : (i + 1) * m2, j * n2 : (j + 1) * n2]
            R[i * n1 + j, :] = block.ravel()

    U, s, Vt = np.linalg.svd(R, full_matrices=False)

    A = (np.sqrt(s[0]) * U[:, 0]).reshape(m1, n1)
    B = (np.sqrt(s[0]) * Vt[0, :]).reshape(m2, n2)

    approx = np.kron(A, B)
    error = float(np.linalg.norm(C - approx, "fro"))
    rel_error = error / max(np.linalg.norm(C, "fro"), 1e-30)

    return DescriptiveResult(
        name="kronecker_decompose",
        value={"A": A, "B": B},
        extra={
            "m1": m1,
            "n1": n1,
            "m2": m2,
            "n2": n2,
            "fro_error": error,
            "rel_error": rel_error,
            "sigma_0": float(s[0]),
        },
    )


megtr = kronecker_decompose


def cheatsheet() -> str:
    return "Knowing yourself is the beginning of all wisdom. — Aristotle"
