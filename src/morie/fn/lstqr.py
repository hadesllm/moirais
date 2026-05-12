# morie.fn -- function file (hadesllm/morie)
"""Least squares via QR decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lstsq_qr(
    A: np.ndarray,
    b: np.ndarray,
) -> DescriptiveResult:
    """Least squares solution via QR decomposition.

    Solves min ||Ax - b||_2 by factoring A = QR and solving Rx = Q^T b.

    Parameters
    ----------
    A : ndarray
        Design matrix (m x n), m >= n.
    b : ndarray
        Observation vector of length m.

    Returns
    -------
    DescriptiveResult
        ``value`` is the residual norm; ``extra`` has x (coefficients).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape
    Q, R = np.linalg.qr(A, mode="reduced")
    x = np.linalg.solve(R, Q.T @ b)
    residual = float(np.linalg.norm(A @ x - b))
    return DescriptiveResult(
        name="Least Squares (QR)",
        value=residual,
        extra={"x": x, "Q": Q, "R": R},
    )


lstqr = lstsq_qr
