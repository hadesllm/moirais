# moirais.fn — function file (hadesllm/moirais)
"""Gaussian mixture model clustering (EM)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gmm_cluster(
    data: np.ndarray,
    n_components: int = 3,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> DescriptiveResult:
    """Gaussian Mixture Model via Expectation-Maximisation.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    n_components : int
        Number of mixture components.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Log-likelihood convergence tolerance.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is hard cluster labels.
        ``extra`` has ``responsibilities``, ``means``, ``weights``,
        ``log_likelihood``, ``bic``, ``aic``.
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape
    k = n_components
    rng = np.random.default_rng(seed)

    idx = rng.choice(n, size=k, replace=False)
    means = X[idx].copy()
    covs = [np.eye(p) for _ in range(k)]
    weights = np.ones(k) / k

    ll_prev = -np.inf

    for _ in range(max_iter):
        resp = np.zeros((n, k))
        for j in range(k):
            diff = X - means[j]
            cov_j = covs[j] + np.eye(p) * 1e-6
            cov_inv = np.linalg.inv(cov_j)
            _, logdet = np.linalg.slogdet(cov_j)
            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            resp[:, j] = np.log(weights[j]) - 0.5 * logdet - 0.5 * p * np.log(2 * np.pi) + exponent

        log_resp_max = resp.max(axis=1, keepdims=True)
        resp_norm = np.exp(resp - log_resp_max)
        resp_sum = resp_norm.sum(axis=1, keepdims=True)
        resp_norm /= resp_sum

        ll = float(np.sum(np.log(resp_sum.ravel()) + log_resp_max.ravel()))
        if abs(ll - ll_prev) < tol:
            break
        ll_prev = ll

        Nk = resp_norm.sum(axis=0)
        for j in range(k):
            weights[j] = Nk[j] / n
            means[j] = (resp_norm[:, j:j+1].T @ X).ravel() / max(Nk[j], 1e-12)
            diff = X - means[j]
            covs[j] = (diff.T * resp_norm[:, j]) @ diff / max(Nk[j], 1e-12)

    labels = np.argmax(resp_norm, axis=1)
    n_params = k * (1 + p + p * (p + 1) / 2) - 1
    bic = -2 * ll + n_params * np.log(n)
    aic = -2 * ll + 2 * n_params

    return DescriptiveResult(
        name="GMM",
        value=labels,
        extra={
            "responsibilities": resp_norm,
            "means": means,
            "weights": weights,
            "log_likelihood": ll,
            "bic": float(bic),
            "aic": float(aic),
        },
    )


gmmcl = gmm_cluster


def cheatsheet() -> str:
    return "gmm_cluster({}) -> Gaussian mixture model clustering (EM)."
