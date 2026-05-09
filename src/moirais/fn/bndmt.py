# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Banded matrix solver."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def banded_solve(
    A: np.ndarray,
    b: np.ndarray,
) -> DescriptiveResult:
    """Solve a banded linear system Ax = b.

    Detects the bandwidth of *A* automatically and uses numpy's solver.

    Parameters
    ----------
    A : ndarray
        Banded square matrix (n x n).
    b : ndarray
        Right-hand side vector.

    Returns
    -------
    DescriptiveResult
        ``value`` is the bandwidth; ``extra`` has x (solution).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[0]
    bw = 0
    for i in range(n):
        for j in range(n):
            if abs(A[i, j]) > 1e-15:
                bw = max(bw, abs(i - j))
    x = np.linalg.solve(A, b)
    return DescriptiveResult(
        name="Banded Solve",
        value=int(bw),
        extra={"x": x, "bandwidth": int(bw)},
    )


bndmt = banded_solve
