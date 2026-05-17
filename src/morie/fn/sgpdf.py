"""Positive definiteness check for covariance matrices."""

from __future__ import annotations

from ._containers import DescriptiveResult


def positive_definiteness_check(C_matrix):
    """Check whether a covariance matrix is positive (semi-)definite.

    Computes eigenvalues and reports the minimum.

    .. epigraph:: No man ever steps in the same river twice. -- Heraclitus

    Parameters
    ----------
    C_matrix : array_like
        Square covariance matrix.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    C = np.asarray(C_matrix, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C_matrix must be square")

    eigenvalues = np.linalg.eigvalsh(C)
    min_eig = float(eigenvalues.min())
    is_pd = bool(min_eig > 0)
    is_psd = bool(min_eig >= -1e-10)

    return DescriptiveResult(
        name="positive_definiteness_check",
        value=min_eig,
        extra={
            "min_eigenvalue": min_eig,
            "max_eigenvalue": float(eigenvalues.max()),
            "is_positive_definite": is_pd,
            "is_positive_semidefinite": is_psd,
            "n_negative": int(np.sum(eigenvalues < -1e-10)),
            "eigenvalues": eigenvalues.tolist(),
        },
    )


sgpdf = positive_definiteness_check


def cheatsheet() -> str:
    return "positive_definiteness_check({}) -> Positive definiteness check for covariance matrices."
