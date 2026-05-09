# moirais.fn — function file (hadesllm/moirais)
"""Knowing yourself is the beginning of all wisdom. — Aristotle"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def qr_factorize(
    A: np.ndarray,
) -> DescriptiveResult:
    """
    QR decomposition via Householder reflections.

    Decomposes A = QR where Q is orthogonal and R is upper triangular.

    :param A: Matrix of shape (m, n) with m >= n.
    :return: DescriptiveResult with Q, R matrices.
    :raises ValueError: If A has more columns than rows.

    References
    ----------
    Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*.
    4th ed. Johns Hopkins University Press. Ch. 5.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix.")
    m, n = A.shape
    if m < n:
        raise ValueError(f"Need m >= n, got {m} x {n}.")

    R = A.copy()
    Q = np.eye(m)

    for j in range(min(m - 1, n)):
        x = R[j:, j].copy()
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x)
        if x[0] < 0:
            e1[0] = -e1[0]
        v = x + e1
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-15:
            continue
        v = v / v_norm

        R[j:, j:] -= 2.0 * np.outer(v, v @ R[j:, j:])
        Q[:, j:] -= 2.0 * np.outer(Q[:, j:] @ v, v)

    orthogonality_error = float(np.max(np.abs(Q.T @ Q - np.eye(m))))

    return DescriptiveResult(
        name="QR Factorization",
        value=float(np.abs(np.prod(np.diag(R[:n, :n])))),
        extra={
            "Q": Q,
            "R": R,
            "shape": (m, n),
            "orthogonality_error": orthogonality_error,
        },
    )


short = qr_factorize


def cheatsheet() -> str:
    return "qr_factorize({}) -> QR factorization via Householder. 'I am your father.' -- Dar"
