# moirais.fn — function file (hadesllm/moirais)
"""Kronecker product and related operations."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def kronecker(
    A: np.ndarray,
    B: np.ndarray,
) -> DescriptiveResult:
    """Kronecker (tensor) product of two matrices.

    Computes A (x) B, resulting in an (m*p) x (n*q) matrix where
    A is m x n and B is p x q.

    Parameters
    ----------
    A : ndarray
        First matrix.
    B : ndarray
        Second matrix.

    Returns
    -------
    DescriptiveResult
        ``value`` is the number of elements; ``extra`` has the result matrix.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    K = np.kron(A, B)
    return DescriptiveResult(
        name="Kronecker Product",
        value=K.size,
        extra={"matrix": K, "shape": K.shape},
    )


krncr = kronecker
