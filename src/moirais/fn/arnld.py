# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Arnoldi iteration for non-symmetric eigenvalue problems."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def arnoldi(
    A: np.ndarray,
    *,
    k: int = 5,
    seed: int = 42,
) -> DescriptiveResult:
    """Arnoldi iteration building an orthonormal Krylov basis.

    Produces Q (n x k) and H (k x k upper Hessenberg) such that
    AQ = QH + residual.  Eigenvalues of H approximate those of A.

    Parameters
    ----------
    A : ndarray
        Square matrix (n x n).
    k : int
        Number of Arnoldi steps.
    seed : int
        Random seed for starting vector.

    Returns
    -------
    DescriptiveResult
        ``extra`` has Q, H, and Ritz values.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    k = min(k, n)
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(n)
    q = q / np.linalg.norm(q)
    Q = np.zeros((n, k))
    H = np.zeros((k, k))
    Q[:, 0] = q
    for j in range(k):
        v = A @ Q[:, j]
        for i in range(j + 1):
            H[i, j] = Q[:, i] @ v
            v -= H[i, j] * Q[:, i]
        if j < k - 1:
            H[j + 1, j] = np.linalg.norm(v)
            if H[j + 1, j] < 1e-14:
                break
            Q[:, j + 1] = v / H[j + 1, j]
    ritz = np.sort(np.abs(np.linalg.eigvals(H)))[::-1]
    return DescriptiveResult(
        name="Arnoldi",
        value=float(ritz[0]),
        extra={"Q": Q, "H": H, "ritz_values": ritz},
    )


arnld = arnoldi
