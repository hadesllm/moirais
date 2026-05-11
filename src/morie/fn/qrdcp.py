# morie.fn — function file (hadesllm/morie)
"""QR decomposition via Householder reflections."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def qr_decomposition(
    A: np.ndarray,
) -> DescriptiveResult:
    """QR decomposition via Householder reflections.

    Factors *A* (m x n, m >= n) into Q (orthogonal) and R (upper triangular)
    such that A = QR.

    Parameters
    ----------
    A : ndarray
        Matrix of shape (m, n).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains Q and R arrays.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    for k in range(min(m - 1, n)):
        x = R[k:, k].copy()
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) * (1 if x[0] >= 0 else -1)
        v = x + e1
        v = v / (np.linalg.norm(v) + 1e-15)
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)
    return DescriptiveResult(
        name="QR Decomposition (Householder)",
        value=None,
        extra={"Q": Q, "R": R},
    )


qrdcp = qr_decomposition
