"""Singular value decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def svd_compute(
    A: np.ndarray,
    *,
    full_matrices: bool = True,
) -> DescriptiveResult:
    """Singular Value Decomposition.

    Factors A = U diag(S) V^T.

    Parameters
    ----------
    A : ndarray
        Input matrix (m x n).
    full_matrices : bool
        If True return full U/Vt; otherwise economy-size.

    Returns
    -------
    DescriptiveResult
        ``value`` is the numerical rank; ``extra`` has U, S, Vt.
    """
    A = np.asarray(A, dtype=float)
    U, S, Vt = np.linalg.svd(A, full_matrices=full_matrices)
    rank = int(np.sum(S[0] * max(A.shape) * np.finfo(float).eps < S))
    return DescriptiveResult(
        name="SVD",
        value=rank,
        extra={"U": U, "S": S, "Vt": Vt},
    )


svdcp = svd_compute
