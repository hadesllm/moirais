# morie.fn — function file (hadesllm/morie)
"""Hadamard (element-wise) product."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def hadamard_product(
    A: np.ndarray,
    B: np.ndarray,
) -> DescriptiveResult:
    """Hadamard (element-wise) product of two matrices.

    Parameters
    ----------
    A : ndarray
        First matrix (m x n).
    B : ndarray
        Second matrix (m x n), same shape as A.

    Returns
    -------
    DescriptiveResult
        ``value`` is the Frobenius norm of the result; ``extra`` has the
        result matrix.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
    H = A * B
    return DescriptiveResult(
        name="Hadamard Product",
        value=float(np.linalg.norm(H, "fro")),
        extra={"matrix": H},
    )


hadpr = hadamard_product
