# morie.fn — function file (hadesllm/morie)
"""Cholesky decomposition."""

import numpy as np

from ._containers import DescriptiveResult


def cholesky_decompose(matrix: np.ndarray) -> DescriptiveResult:
    """
    Cholesky decomposition of a positive-definite matrix.

    Computes lower triangular L such that A = L L^T.

    :param matrix: (n, n) symmetric positive-definite matrix.
    :return: DescriptiveResult with L in extra.
    :raises ValueError: If matrix not positive definite.

    References
    ----------
    Golub GH, Van Loan CF (2013). Matrix Computations. 4th ed.
    Johns Hopkins University Press.
    """
    A = np.asarray(matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is not positive definite.")
    log_det = 2 * np.sum(np.log(np.diag(L)))
    return DescriptiveResult(
        name="cholesky",
        value=float(log_det),
        extra={"L": L, "log_determinant": float(log_det), "n": A.shape[0]},
    )


chol = cholesky_decompose


def cheatsheet() -> str:
    return "cholesky_decompose({}) -> Cholesky decomposition."
