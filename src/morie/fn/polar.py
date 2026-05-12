# morie.fn -- function file (hadesllm/morie)
"""Polar decomposition A = UP. 'There is another.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def polar_decompose(
    A: np.ndarray,
) -> DescriptiveResult:
    """
    Compute the polar decomposition A = U P.

    U is unitary (orthogonal for real matrices) and P is symmetric
    positive semi-definite. Computed via SVD: A = W S V^T, then
    U = W V^T and P = V S V^T.

    :param A: Matrix of shape (m, n) with m >= n.
    :return: DescriptiveResult with U and P matrices.
    :raises ValueError: If A is not 2D.

    References
    ----------
    Higham, N. J. (1986). Computing the polar decomposition -- with
    applications. *SIAM J. Sci. Stat. Comput.*, 7(4), 1160-1174.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix.")

    W, S, Vt = np.linalg.svd(A, full_matrices=False)
    U = W @ Vt
    P = Vt.T @ np.diag(S) @ Vt

    reconstruction_error = float(np.max(np.abs(A - U @ P)))
    symmetry_error = float(np.max(np.abs(P - P.T)))

    eigvals_P = np.linalg.eigvalsh(P)

    return DescriptiveResult(
        name="Polar Decomposition",
        value=float(np.linalg.det(U)) if A.shape[0] == A.shape[1] else float(S[0]),
        extra={
            "U": U,
            "P": P,
            "singular_values": S,
            "reconstruction_error": reconstruction_error,
            "symmetry_error": symmetry_error,
            "P_eigenvalues": eigvals_P,
            "P_positive_semidefinite": bool(np.all(eigvals_P >= -1e-10)),
            "shape": A.shape,
        },
    )


short = polar_decompose


def cheatsheet() -> str:
    return "polar_decompose({}) -> Polar decomposition A = UP. 'There is another.'"
