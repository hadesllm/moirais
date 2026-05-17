# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""A-M eigensolve for stimulus positions."""

from __future__ import annotations

from ._containers import DescriptiveResult


def am_eigensolve(M) -> DescriptiveResult:
    """Eigen-decomposition of centered A-M matrix.

    :param M: Centered matrix from am_matrix_setup.
    :return: DescriptiveResult with eigenvalues and eigenvectors.

    .. epigraph:: No man ever steps in the same river twice. -- Heraclitus
    """
    import numpy as np

    M = np.asarray(M, dtype=float)
    MtM = M.T @ M
    eigvals, eigvecs = np.linalg.eigh(MtM)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return DescriptiveResult(
        name="am_eigensolve",
        value=float(eigvals[0]),
        extra={
            "eigenvalues": eigvals.tolist(),
            "eigenvectors": eigvecs.tolist(),
            "n_dims": len(eigvals),
        },
    )


ameig = am_eigensolve


def cheatsheet() -> str:
    return "am_eigensolve({}) -> A-M eigensolve for stimulus positions."
