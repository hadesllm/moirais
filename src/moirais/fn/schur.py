# moirais.fn — function file (hadesllm/moirais)
"""No man ever steps in the same river twice. — Heraclitus"""

from __future__ import annotations

import numpy as np
from scipy import linalg as la

from ._containers import DescriptiveResult


def schur_decompose(
    A: np.ndarray,
) -> DescriptiveResult:
    """
    Compute the Schur decomposition A = Q T Q^H.

    T is upper quasi-triangular (real Schur form) with eigenvalues on
    or near the diagonal. Q is unitary.

    :param A: Square matrix.
    :return: DescriptiveResult with T, Q matrices and eigenvalues.
    :raises ValueError: If A is not square.

    References
    ----------
    Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*.
    4th ed. Johns Hopkins University Press. Ch. 7.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    T, Q = la.schur(A, output="real")
    eigenvalues = np.linalg.eigvals(A)

    reconstruction_error = float(np.max(np.abs(A - Q @ T @ Q.T)))

    return DescriptiveResult(
        name="Schur Decomposition",
        value=float(np.max(np.abs(eigenvalues))),
        extra={
            "T": T,
            "Q": Q,
            "eigenvalues": eigenvalues,
            "spectral_radius": float(np.max(np.abs(eigenvalues))),
            "reconstruction_error": reconstruction_error,
            "n": A.shape[0],
        },
    )


short = schur_decompose


def cheatsheet() -> str:
    return "schur_decompose({}) -> Schur decomposition. 'You underestimate my power.' -- Anakin"
