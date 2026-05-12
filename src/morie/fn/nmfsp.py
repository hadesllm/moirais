# morie.fn -- function file (hadesllm/morie)
"""Sparse Non-negative Matrix Factorization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "There is always a bigger fish."


def nmf_sparse(
    X,
    n_components: int = 5,
    alpha: float = 0.1,
    max_iter: int = 200,
    tol: float = 1e-4,
    seed: int | None = None,
    **kwargs,
) -> DescriptiveResult:
    """Sparse NMF with L1 penalty via multiplicative updates.

    Factorises X ~ W H subject to non-negativity and sparsity on H.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Non-negative input matrix.
    n_components : int
        Number of components (default 5).
    alpha : float
        L1 sparsity penalty on H (default 0.1).
    max_iter : int
        Maximum iterations (default 200).
    tol : float
        Convergence tolerance (default 1e-4).
    seed : int or None
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is relative reconstruction error; ``extra`` has
        ``W``, ``H``, ``n_components``.

    References
    ----------
    Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative
    matrix factorization. *NIPS*, 556-562.
    """
    X = np.asarray(X, dtype=float)
    X = np.maximum(X, 0.0)
    m, n = X.shape
    rng = np.random.default_rng(seed)
    W = np.abs(rng.standard_normal((m, n_components))) + 0.1
    H = np.abs(rng.standard_normal((n_components, n))) + 0.1
    eps = 1e-10
    for _ in range(max_iter):
        WH = W @ H + eps
        H_new = H * (W.T @ (X / WH)) / (np.sum(W, axis=0, keepdims=True).T + alpha + eps)
        H = np.maximum(H_new, eps)
        WH = W @ H + eps
        W_new = W * ((X / WH) @ H.T) / (np.sum(H, axis=1, keepdims=True).T + eps)
        W = np.maximum(W_new, eps)
        err = np.linalg.norm(X - W @ H, "fro") / (np.linalg.norm(X, "fro") + eps)
        if err < tol:
            break
    rel_err = float(np.linalg.norm(X - W @ H, "fro") / (np.linalg.norm(X, "fro") + eps))
    return DescriptiveResult(
        name="nmf_sparse",
        value=rel_err,
        extra={"W": W, "H": H, "n_components": n_components},
    )


nmfsp = nmf_sparse


def cheatsheet() -> str:
    return "nmf_sparse({}) -> Sparse Non-negative Matrix Factorization."
