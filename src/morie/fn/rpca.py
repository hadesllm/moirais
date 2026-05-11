# morie.fn — function file (hadesllm/morie)
"""Robust PCA via ADMM (low-rank + sparse decomposition)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Power! Unlimited power!"


def robust_pca(X, lambda_: float | None = None, max_iter: int = 200, tol: float = 1e-7, **kwargs) -> DescriptiveResult:
    """Robust PCA: decompose X = L + S (low-rank + sparse) via ADMM.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Input data matrix.
    lambda_ : float or None
        Sparsity weight. Default: 1/sqrt(max(m,n)).
    max_iter : int
        Maximum ADMM iterations (default 200).
    tol : float
        Convergence tolerance (default 1e-7).

    Returns
    -------
    DescriptiveResult
        ``value`` is rank of L; ``extra`` has ``L`` (low-rank), ``S``
        (sparse), ``iterations``.

    References
    ----------
    Candes, E. J., Li, X., Ma, Y., & Wright, J. (2011). Robust
    principal component analysis? *J. ACM*, 58(3), 1-37.
    """
    X = np.asarray(X, dtype=float)
    m, n = X.shape
    if lambda_ is None:
        lambda_ = 1.0 / np.sqrt(max(m, n))
    mu = m * n / (4.0 * np.sum(np.abs(X)))
    mu_inv = 1.0 / mu

    def _shrink(M, tau):
        return np.sign(M) * np.maximum(np.abs(M) - tau, 0.0)

    def _svd_threshold(M, tau):
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        s_shrunk = np.maximum(s - tau, 0.0)
        return U * s_shrunk @ Vt, int(np.sum(s_shrunk > 0))

    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    iters = 0
    for i in range(max_iter):
        L, rank = _svd_threshold(X - S + mu_inv * Y, mu_inv)
        S = _shrink(X - L + mu_inv * Y, lambda_ * mu_inv)
        Z = X - L - S
        Y = Y + mu * Z
        iters = i + 1
        if np.linalg.norm(Z, "fro") / (np.linalg.norm(X, "fro") + 1e-15) < tol:
            break
    return DescriptiveResult(
        name="robust_pca",
        value=rank,
        extra={"L": L, "S": S, "iterations": iters, "lambda": lambda_},
    )


rpca = robust_pca


def cheatsheet() -> str:
    return "robust_pca({}) -> Robust PCA via ADMM (low-rank + sparse decomposition)."
