# morie.fn — function file (hadesllm/morie)
"""Eigenvalue decomposition for symmetric matrices."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def eigen_symmetric(
    A: np.ndarray,
) -> DescriptiveResult:
    """Eigenvalue decomposition for a real symmetric matrix.

    Uses numpy.linalg.eigh which guarantees real eigenvalues and orthogonal
    eigenvectors for symmetric input.

    Parameters
    ----------
    A : ndarray
        Symmetric matrix (n x n).

    Returns
    -------
    DescriptiveResult
        ``value`` is the largest eigenvalue; ``extra`` has eigenvalues
        (ascending) and eigenvectors.
    """
    A = np.asarray(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    vals, vecs = np.linalg.eigh(A)
    return DescriptiveResult(
        name="Eigen (symmetric)",
        value=float(vals[-1]),
        extra={"eigenvalues": vals, "eigenvectors": vecs},
    )


eigsm = eigen_symmetric
