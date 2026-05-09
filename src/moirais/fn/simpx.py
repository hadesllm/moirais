"""Simplex method for linear programming."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def simplex_lp(
    c: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
) -> DescriptiveResult:
    """Simplex method for linear programming.

    Solves: minimise c^T x subject to A_ub @ x <= b_ub, x >= 0.

    Uses the two-phase simplex with slack variables.

    Parameters
    ----------
    c : ndarray
        Objective coefficients (length n).
    A_ub : ndarray
        Inequality constraint matrix (m x n).
    b_ub : ndarray
        Inequality constraint bounds (length m).

    Returns
    -------
    DescriptiveResult
        ``value`` is the optimal objective; ``extra`` has x.
    """
    c = np.asarray(c, dtype=float)
    A = np.asarray(A_ub, dtype=float)
    b = np.asarray(b_ub, dtype=float)
    m, n = A.shape
    for i in range(m):
        if b[i] < 0:
            A[i] *= -1
            b[i] *= -1
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n] = A
    T[:m, n : n + m] = np.eye(m)
    T[:m, -1] = b
    T[-1, :n] = c
    for _ in range(2 * (m + n)):
        col = np.argmin(T[-1, :-1])
        if T[-1, col] >= -1e-12:
            break
        ratios = np.full(m, np.inf)
        for i in range(m):
            if T[i, col] > 1e-12:
                ratios[i] = T[i, -1] / T[i, col]
        row = np.argmin(ratios)
        if ratios[row] == np.inf:
            raise ValueError("LP is unbounded")
        T[row] /= T[row, col]
        for i in range(m + 1):
            if i != row:
                T[i] -= T[i, col] * T[row]
    x = np.zeros(n)
    for j in range(n):
        col_vals = T[:m, j]
        if np.sum(np.abs(col_vals) > 1e-12) == 1:
            row_idx = np.argmax(np.abs(col_vals))
            if abs(col_vals[row_idx] - 1.0) < 1e-12:
                x[j] = T[row_idx, -1]
    return DescriptiveResult(
        name="Simplex LP",
        value=float(c @ x),
        extra={"x": x},
    )


simpx = simplex_lp
