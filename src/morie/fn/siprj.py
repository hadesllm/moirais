"""Single-index projection pursuit."""

from __future__ import annotations

import numpy as np


def siprj(
    y: np.ndarray,
    X: np.ndarray,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
    max_iter: int = 30,
    tol: float = 1e-5,
) -> dict:
    r"""
    Single-index model via projection pursuit regression.

    Iteratively finds the projection direction :math:`\beta` that
    minimises the residual sum of squares when the link function
    is estimated nonparametrically:

    .. math::

        \hat{\beta} = \arg\min_{\|\beta\|=1}
        \sum_{i=1}^n (Y_i - \hat{G}(X_i'\beta))^2

    Parameters
    ----------
    y : np.ndarray
        Response (n,).
    X : np.ndarray
        Covariates (n, p), p >= 2.
    bandwidth : float or None
        Kernel bandwidth.
    kernel : str
        Kernel function.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict
        ``beta``, ``index``, ``g_hat``, ``rss``, ``n_iter``,
        ``converged``, ``n_obs``.

    References
    ----------
    Friedman, J. & Stuetzle, W. (1981). Projection pursuit regression.
        JASA, 76, 817-823.
    Horowitz (2009). Ch 2.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError(f"y length {y.shape[0]} != X rows {n}.")
    if p < 2:
        raise ValueError("Need p >= 2 covariates.")

    from morie.fn.nwker import _get_kernel, _silverman_bw

    k_fn = _get_kernel(kernel)

    cov = X.T @ X / n
    eigvals, eigvecs = np.linalg.eigh(cov)
    beta = eigvecs[:, -1].copy()
    beta /= np.linalg.norm(beta)

    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        idx = X @ beta
        h = bandwidth if bandwidth is not None else _silverman_bw(idx)

        diff = idx[:, None] - idx[None, :]
        W = k_fn(diff / h)
        np.fill_diagonal(W, 0.0)
        denom = W.sum(axis=1)
        denom = np.where(denom < 1e-15, 1.0, denom)
        g_hat = (W @ y) / denom

        resid = y - g_hat

        diff2 = idx[:, None] - idx[None, :]
        u = diff2 / h
        k_prime = -u * k_fn(u) / h
        np.fill_diagonal(k_prime, 0.0)

        grad = np.zeros(p)
        for j in range(p):
            dg_dbeta = np.zeros(n)
            for i in range(n):
                xdiff = X[:, j] - X[i, j]
                dg_dbeta[i] = (k_prime[i, :] * y).sum() / (denom[i] * h)
            grad[j] = -2 * np.mean(resid * dg_dbeta)

        grad -= (grad @ beta) * beta
        step = 0.1 / (np.linalg.norm(grad) + 1e-15)
        beta_new = beta - step * grad
        beta_new /= np.linalg.norm(beta_new)

        n_iter = iteration + 1
        if np.linalg.norm(beta_new - beta) < tol:
            converged = True
            beta = beta_new
            break
        beta = beta_new

    idx_final = X @ beta
    h = bandwidth if bandwidth is not None else _silverman_bw(idx_final)
    diff = idx_final[:, None] - idx_final[None, :]
    W = k_fn(diff / h)
    np.fill_diagonal(W, 0.0)
    denom = W.sum(axis=1)
    denom = np.where(denom < 1e-15, 1.0, denom)
    g_hat = (W @ y) / denom
    rss = float(np.sum((y - g_hat) ** 2))

    return {
        "beta": beta.tolist(),
        "index": idx_final.tolist(),
        "g_hat": g_hat.tolist(),
        "rss": rss,
        "n_iter": n_iter,
        "converged": converged,
        "n_obs": n,
    }


siprj_fn = siprj


def cheatsheet() -> str:
    return "siprj({y, X}) -> Single-index via projection pursuit."
