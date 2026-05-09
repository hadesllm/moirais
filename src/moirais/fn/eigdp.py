# moirais.fn — function file (hadesllm/moirais)
"""Eigendecomposition."""

import numpy as np

from ._containers import DescriptiveResult
def eigen_decompose(A, **kwargs) -> DescriptiveResult:
    """
    Compute eigenvalues and eigenvectors of a square matrix.

    .. math::

        A v = \\lambda v

    :param A: (d, d) square matrix.
    :return: DescriptiveResult with eigenvalues and eigenvectors.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    sym = np.allclose(A, A.T)
    if sym:
        eigvals, eigvecs = np.linalg.eigh(A)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
    else:
        eigvals, eigvecs = np.linalg.eig(A)
        idx = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
    return DescriptiveResult(
        name="eigen_decompose",
        value=float(np.max(np.abs(eigvals))),
        extra={
            "eigenvalues": eigvals.tolist(),
            "symmetric": sym,
            "dim": A.shape[0],
            "rank": int(np.linalg.matrix_rank(A)),
        },
    )


eigdp = eigen_decompose


def cheatsheet() -> str:
    return "eigen_decompose({}) -> Eigendecomposition."
