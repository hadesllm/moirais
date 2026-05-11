# morie.fn — function file (hadesllm/morie)
"""Finite mixture model (general Gaussian)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def finite_mixture(
    X: np.ndarray,
    n_components: int = 2,
    *,
    family: str = "gaussian",
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int | None = None,
) -> DescriptiveResult:
    """Finite Gaussian mixture model via EM (full covariance).

    Parameters
    ----------
    X : (n,) or (n, p) array
    n_components : int
    family : str
        Currently only 'gaussian'.
    max_iter, tol : convergence
    seed : int, optional

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    K = n_components
    rng = np.random.default_rng(seed)

    pi = np.ones(K) / K
    idx = rng.choice(n, K, replace=False)
    mu = X[idx].copy()
    covs = [np.eye(p) * np.var(X, axis=0).mean() for _ in range(K)]

    ll_old = -np.inf
    for _ in range(max_iter):
        log_resp = np.zeros((n, K))
        for k in range(K):
            diff = X - mu[k]
            try:
                L = np.linalg.cholesky(covs[k] + 1e-6 * np.eye(p))
                log_det = 2 * np.sum(np.log(np.diag(L)))
                sol = np.linalg.solve(L, diff.T)
                mah = np.sum(sol**2, axis=0)
            except np.linalg.LinAlgError:
                mah = np.sum(diff**2, axis=1)
                log_det = 0.0
            log_resp[:, k] = -0.5 * (mah + log_det + p * np.log(2 * np.pi)) + np.log(pi[k] + 1e-12)

        mx = log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp - mx)
        resp /= resp.sum(axis=1, keepdims=True)

        ll_new = float(np.sum(np.log(np.sum(np.exp(log_resp - mx), axis=1)) + mx.ravel()))
        if abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

        Nk = resp.sum(axis=0) + 1e-12
        pi = Nk / n
        for k in range(K):
            mu[k] = resp[:, k] @ X / Nk[k]
            diff = X - mu[k]
            covs[k] = (diff.T * resp[:, k]) @ diff / Nk[k] + 1e-6 * np.eye(p)

    classes = resp.argmax(axis=1)
    n_params = K - 1 + K * p + K * p * (p + 1) / 2
    bic = -2 * ll_new + n_params * np.log(n)

    return DescriptiveResult(
        name="fmm",
        value=float(bic),
        extra={
            "n_components": K,
            "family": family,
            "log_likelihood": float(ll_new),
            "class_sizes": [int(np.sum(classes == k)) for k in range(K)],
            "n": n,
            "p": p,
        },
    )


fmm = finite_mixture


def cheatsheet() -> str:
    return "finite_mixture({}) -> Finite mixture model (general Gaussian)."
