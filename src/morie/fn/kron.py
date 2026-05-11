# morie.fn — function file (hadesllm/morie)
"""Kronecker product of two matrices."""

import numpy as np

from ._containers import DescriptiveResult


def kronecker_product(A: np.ndarray, B: np.ndarray) -> DescriptiveResult:
    """
    Compute the Kronecker product A (x) B.

    :param A: (m, n) first matrix.
    :param B: (p, q) second matrix.
    :return: DescriptiveResult with (mp, nq) Kronecker product.

    References
    ----------
    Graham A (1981). Kronecker Products and Matrix Calculus with
    Applications. Ellis Horwood.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    K = np.kron(A, B)
    return DescriptiveResult(
        name="kronecker_product",
        value=float(K.shape[0] * K.shape[1]),
        extra={"product": K, "shape": K.shape, "A_shape": A.shape, "B_shape": B.shape},
    )


kron = kronecker_product


def cheatsheet() -> str:
    return "kronecker_product({}) -> Kronecker product of two matrices."
