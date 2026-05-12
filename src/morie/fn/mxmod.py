# morie.fn -- function file (hadesllm/morie)
"""Semiparametric mixture model (EM with nonparametric component)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def mxmod(
    data: np.ndarray,
    *,
    n_components: int = 2,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> dict:
    r"""
    Semiparametric Gaussian mixture model via EM.

    Fits a finite mixture of Gaussians to univariate data using the
    Expectation-Maximization algorithm.  While each component is
    parametric (Gaussian), the overall density

    .. math::

        f(x) = \sum_{k=1}^{K} \pi_k \, \phi(x; \mu_k, \sigma_k^2)

    is a flexible semiparametric approximation to an arbitrary
    distribution (Lindsay, 1995).

    The EM iterates:

    **E-step:**

    .. math::

        \gamma_{ik} = \frac{\pi_k \, \phi(x_i; \mu_k, \sigma_k^2)}
        {\sum_{j=1}^{K} \pi_j \, \phi(x_i; \mu_j, \sigma_j^2)}

    **M-step:**

    .. math::

        \pi_k = \frac{1}{n} \sum_i \gamma_{ik}, \quad
        \mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}, \quad
        \sigma_k^2 = \frac{\sum_i \gamma_{ik}(x_i - \mu_k)^2}{\sum_i \gamma_{ik}}

    :param data: Univariate data array, shape ``(n,)``.
    :param n_components: Number of mixture components *K*. Default 2.
    :param max_iter: Maximum EM iterations. Default 200.
    :param tol: Convergence tolerance on log-likelihood change. Default 1e-6.
    :param seed: Random seed for initialization. Default 42.
    :return: dict with ``means``, ``variances``, ``weights``,
        ``log_likelihood``, ``n_iter``, ``bic``, ``converged``.
    :raises ValueError: If *n_components* < 1 or data is empty.

    References
    ----------
    Dempster, A. P., Laird, N. M. & Rubin, D. B. (1977). Maximum
        likelihood from incomplete data via the EM algorithm. *JRSS-B*,
        39(1), 1--38.
    Lindsay, B. G. (1995). *Mixture Models: Theory, Geometry and
        Applications*. IMS.
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
        in Econometrics*. Springer, Ch. 5.
    """
    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n == 0:
        raise ValueError("data must not be empty.")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}.")

    rng = np.random.default_rng(seed)
    K = n_components

    idx = rng.choice(n, size=K, replace=False)
    means = data[idx].copy()
    variances = np.full(K, np.var(data) / K)
    weights = np.full(K, 1.0 / K)

    log_lik = -np.inf
    converged = False
    n_iter = 0

    for it in range(max_iter):
        resp = np.empty((n, K))
        for k in range(K):
            sd = max(np.sqrt(variances[k]), 1e-12)
            resp[:, k] = weights[k] * norm.pdf(data, loc=means[k], scale=sd)
        row_sum = resp.sum(axis=1, keepdims=True)
        row_sum = np.maximum(row_sum, 1e-300)
        resp /= row_sum

        new_ll = float(np.sum(np.log(row_sum.ravel())))

        n_k = resp.sum(axis=0)
        for k in range(K):
            if n_k[k] < 1e-12:
                continue
            weights[k] = n_k[k] / n
            means[k] = np.sum(resp[:, k] * data) / n_k[k]
            variances[k] = np.sum(resp[:, k] * (data - means[k]) ** 2) / n_k[k]
            variances[k] = max(variances[k], 1e-12)

        n_iter = it + 1
        if abs(new_ll - log_lik) < tol:
            converged = True
            log_lik = new_ll
            break
        log_lik = new_ll

    n_params = 3 * K - 1
    bic = -2 * log_lik + n_params * np.log(n)

    return {
        "means": means,
        "variances": variances,
        "weights": weights,
        "log_likelihood": log_lik,
        "n_iter": n_iter,
        "bic": float(bic),
        "converged": converged,
    }


mxmod_fn = mxmod


def cheatsheet() -> str:
    return "mxmod(data) -> Semiparametric mixture model (EM)."
