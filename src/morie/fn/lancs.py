# morie.fn — function file (hadesllm/morie)
"""Lanczos algorithm for largest eigenvalues of symmetric matrices."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lanczos(
    A: np.ndarray,
    *,
    k: int = 5,
    seed: int = 42,
) -> DescriptiveResult:
    """Lanczos algorithm for the *k* largest eigenvalues of a symmetric matrix.

    Builds a tridiagonal matrix via Lanczos iteration and extracts its
    eigenvalues as approximations to those of *A*.

    Parameters
    ----------
    A : ndarray
        Symmetric matrix (n x n).
    k : int
        Number of Lanczos steps (and approximate eigenvalues).
    seed : int
        Random seed for starting vector.

    Returns
    -------
    DescriptiveResult
        ``value`` is the largest eigenvalue; ``extra`` has all *k* Ritz values.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    k = min(k, n)
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(n)
    q = q / np.linalg.norm(q)
    alpha = np.zeros(k)
    beta = np.zeros(k)
    Q = np.zeros((n, k))
    Q[:, 0] = q
    for j in range(k):
        v = A @ Q[:, j]
        if j > 0:
            v -= beta[j] * Q[:, j - 1]
        alpha[j] = Q[:, j] @ v
        v -= alpha[j] * Q[:, j]
        if j < k - 1:
            beta[j + 1] = np.linalg.norm(v)
            if beta[j + 1] < 1e-14:
                k = j + 1
                break
            Q[:, j + 1] = v / beta[j + 1]
    T = np.diag(alpha[:k]) + np.diag(beta[1:k], 1) + np.diag(beta[1:k], -1)
    ritz = np.sort(np.linalg.eigvalsh(T))[::-1]
    return DescriptiveResult(
        name="Lanczos",
        value=float(ritz[0]),
        extra={"ritz_values": ritz, "steps": k},
    )


lancs = lanczos
