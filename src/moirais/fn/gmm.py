# moirais.fn — function file (hadesllm/moirais)
"""Gaussian mixture model via EM algorithm."""

import numpy as np

from ._containers import DescriptiveResult


def gaussian_mixture(X: np.ndarray, n_components: int = 2, n_iter: int = 100, seed: int = 42) -> DescriptiveResult:
    """
    Gaussian Mixture Model (GMM) via Expectation-Maximisation.

    :param X: (n, p) data matrix.
    :param n_components: Number of Gaussian components.
    :param n_iter: Maximum EM iterations.
    :param seed: Random seed.
    :return: DescriptiveResult with labels, means, covariances.

    References
    ----------
    Dempster AP, Laird NM, Rubin DB (1977). Maximum likelihood from
    incomplete data via the EM algorithm. JRSS-B, 39(1), 1-22.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    K = n_components
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, K, replace=False)
    means = X[idx].copy()
    covs = [np.eye(p) for _ in range(K)]
    weights = np.ones(K) / K
    resp = np.zeros((n, K))
    for _ in range(n_iter):
        for k in range(K):
            diff = X - means[k]
            cov_inv = np.linalg.pinv(covs[k])
            sign, logdet = np.linalg.slogdet(covs[k])
            log_norm = -0.5 * (p * np.log(2 * np.pi) + logdet)
            log_exp = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            resp[:, k] = np.log(weights[k] + 1e-300) + log_norm + log_exp
        log_sum = np.max(resp, axis=1, keepdims=True)
        resp = np.exp(resp - log_sum)
        resp /= resp.sum(axis=1, keepdims=True)
        Nk = resp.sum(axis=0)
        for k in range(K):
            if Nk[k] < 1e-10:
                continue
            means[k] = (resp[:, k : k + 1].T @ X).ravel() / Nk[k]
            diff = X - means[k]
            covs[k] = (resp[:, k : k + 1] * diff).T @ diff / Nk[k]
            covs[k] += 1e-6 * np.eye(p)
        weights = Nk / n
    labels = np.argmax(resp, axis=1)
    ll = 0.0
    for k in range(K):
        diff = X - means[k]
        cov_inv = np.linalg.pinv(covs[k])
        _, logdet = np.linalg.slogdet(covs[k])
        ll += np.sum(resp[:, k] * (-0.5 * (p * np.log(2 * np.pi) + logdet + np.sum(diff @ cov_inv * diff, axis=1))))
    return DescriptiveResult(
        name="gaussian_mixture",
        value=float(ll),
        extra={
            "labels": labels,
            "means": means,
            "covariances": covs,
            "weights": weights,
            "loglik": float(ll),
            "n_components": K,
            "n": n,
        },
    )


gmm = gaussian_mixture


def cheatsheet() -> str:
    return "gaussian_mixture({}) -> Gaussian mixture model via EM algorithm."
