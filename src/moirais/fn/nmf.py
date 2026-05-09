# moirais.fn — function file (hadesllm/moirais)
"""Non-negative Matrix Factorization. 'You have power over your mind — not outside events. Realize this, and you will find strength. — Marcus Aurelius' -- Ahsoka Tano"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def nmf(
    X: np.ndarray,
    n_components: int = 5,
    max_iter: int = 200,
    tol: float = 1e-4,
    seed: int = 42,
) -> DescriptiveResult:
    r"""Non-negative Matrix Factorization using multiplicative updates.

    Factorizes a non-negative matrix :math:`X \approx W H` where
    :math:`W \in \mathbb{R}^{n \times k}` and
    :math:`H \in \mathbb{R}^{k \times m}`.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Non-negative input matrix.
    n_components : int
        Number of components (k).
    max_iter : int
        Maximum number of multiplicative update iterations.
    tol : float
        Convergence tolerance on reconstruction error.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        name='NMF', value=reconstruction error (Frobenius norm),
        extra has 'W', 'H', 'n_components', 'n_iter', 'reconstruction_error'.

    References
    ----------
    Lee, D.D. & Seung, H.S. (2001). Algorithms for Non-negative
    Matrix Factorization. *Advances in NIPS*, 13, 556-562.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)

    if np.any(X < 0):
        raise ValueError("NMF requires non-negative input matrix.")

    n, m = X.shape
    k = n_components

    W = rng.uniform(0.01, 1.0, (n, k))
    H = rng.uniform(0.01, 1.0, (k, m))

    eps = 1e-12
    prev_err = float("inf")

    for it in range(1, max_iter + 1):
        H *= (W.T @ X) / (W.T @ W @ H + eps)
        W *= (X @ H.T) / (W @ H @ H.T + eps)

        if it % 10 == 0:
            err = float(np.linalg.norm(X - W @ H, "fro"))
            if abs(prev_err - err) < tol:
                break
            prev_err = err

    final_err = float(np.linalg.norm(X - W @ H, "fro"))

    return DescriptiveResult(
        name="NMF",
        value=final_err,
        extra={
            "W": W,
            "H": H,
            "n_components": k,
            "n_iter": it,
            "reconstruction_error": final_err,
        },
    )


def cheatsheet() -> str:
    return "nmf({}) -> Non-negative Matrix Factorization. 'You have power over your mind — not outside events. Realize this, and you will find strength. — Marcus Aurelius' -- Ahsoka"
