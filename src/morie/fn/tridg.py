"""Tridiagonal matrix solver (Thomas algorithm)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def thomas_solve(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> DescriptiveResult:
    """Thomas algorithm for tridiagonal systems.

    Solves Mx = d where M is tridiagonal with sub-diagonal *a*,
    main diagonal *b*, and super-diagonal *c*.

    Parameters
    ----------
    a : ndarray
        Sub-diagonal of length n-1.
    b : ndarray
        Main diagonal of length n.
    c : ndarray
        Super-diagonal of length n-1.
    d : ndarray
        Right-hand side of length n.

    Returns
    -------
    DescriptiveResult
        ``value`` is 0.0 (success); ``extra`` has x (solution vector).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float).copy()
    c = np.asarray(c, dtype=float).copy()
    d = np.asarray(d, dtype=float).copy()
    n = len(b)
    for i in range(1, n):
        if abs(b[i - 1]) < 1e-15:
            raise ValueError(f"Zero pivot at row {i - 1}")
        m = a[i - 1] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return DescriptiveResult(name="Thomas (tridiagonal)", value=0.0, extra={"x": x})


tridg = thomas_solve
