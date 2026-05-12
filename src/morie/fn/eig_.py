# morie.fn -- function file (hadesllm/morie)
"""Eigenvalue/eigenvector analysis."""

import numpy as np

from ._containers import DescriptiveResult


def eigen_analysis(matrix: np.ndarray) -> DescriptiveResult:
    """
    Eigenvalue/eigenvector decomposition of a square matrix.

    :param matrix: (n, n) square matrix.
    :return: DescriptiveResult with eigenvalues and eigenvectors.
    :raises ValueError: If matrix not square.

    References
    ----------
    Golub GH, Van Loan CF (2013). Matrix Computations. 4th ed.
    Johns Hopkins University Press.
    """
    A = np.asarray(matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")
    if np.allclose(A, A.T):
        eigvals, eigvecs = np.linalg.eigh(A)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    else:
        eigvals, eigvecs = np.linalg.eig(A)
        idx = np.argsort(np.abs(eigvals))[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    spectral_radius = float(np.max(np.abs(eigvals)))
    trace = float(np.sum(eigvals))
    det = float(np.prod(eigvals))
    return DescriptiveResult(
        name="eigen_analysis",
        value=spectral_radius,
        extra={
            "eigenvalues": eigvals,
            "eigenvectors": eigvecs,
            "spectral_radius": spectral_radius,
            "trace": trace,
            "determinant": det,
            "n": A.shape[0],
        },
    )


eig_ = eigen_analysis


def cheatsheet() -> str:
    return "eigen_analysis({}) -> Eigenvalue/eigenvector analysis."
