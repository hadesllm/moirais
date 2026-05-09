# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian hierarchical model (2-level normal)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def bayesian_hierarchical(
    group_data: list[Union[list, np.ndarray]],
    *,
    prior_mu0: float = 0.0,
    prior_sigma0: float = 10.0,
    prior_tau_a: float = 1.0,
    prior_tau_b: float = 1.0,
    n_iter: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Two-level Bayesian hierarchical normal model via Gibbs sampling.

    Level 1: y_{ij} | theta_j, sigma^2 ~ N(theta_j, sigma^2)
    Level 2: theta_j | mu, tau^2 ~ N(mu, tau^2)

    :param group_data: List of arrays, one per group.
    :param prior_mu0: Prior mean for grand mean mu.
    :param prior_sigma0: Prior SD for grand mean mu.
    :param prior_tau_a: Shape of IG prior on tau^2.
    :param prior_tau_b: Scale of IG prior on tau^2.
    :param n_iter: Number of Gibbs iterations.
    :param seed: Random seed.
    :return: Dictionary with mu_samples, tau2_samples, theta_samples, sigma2_samples.

    References
    ----------
    Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed., Ch. 5.
    """
    rng = np.random.default_rng(seed)
    groups = [np.asarray(g, dtype=float) for g in group_data]
    J = len(groups)
    n_j = np.array([len(g) for g in groups])
    y_bar = np.array([float(np.mean(g)) for g in groups])
    N = int(np.sum(n_j))

    mu = float(np.mean(y_bar))
    tau2 = 1.0
    sigma2 = 1.0
    theta = y_bar.copy()

    mu_samples = np.empty(n_iter)
    tau2_samples = np.empty(n_iter)
    sigma2_samples = np.empty(n_iter)
    theta_samples = np.empty((n_iter, J))

    for it in range(n_iter):
        for j in range(J):
            prec_j = n_j[j] / sigma2 + 1.0 / tau2
            mean_j = (n_j[j] * y_bar[j] / sigma2 + mu / tau2) / prec_j
            theta[j] = rng.normal(mean_j, 1.0 / np.sqrt(prec_j))

        prec_mu = J / tau2 + 1.0 / prior_sigma0 ** 2
        mean_mu = (np.sum(theta) / tau2 + prior_mu0 / prior_sigma0 ** 2) / prec_mu
        mu = rng.normal(mean_mu, 1.0 / np.sqrt(prec_mu))

        post_a_tau = prior_tau_a + J / 2.0
        post_b_tau = prior_tau_b + 0.5 * np.sum((theta - mu) ** 2)
        tau2 = 1.0 / rng.gamma(post_a_tau, 1.0 / post_b_tau)

        ss = sum(float(np.sum((g - theta[j]) ** 2)) for j, g in enumerate(groups))
        post_a_sig = 1.0 + N / 2.0
        post_b_sig = 1.0 + ss / 2.0
        sigma2 = 1.0 / rng.gamma(post_a_sig, 1.0 / post_b_sig)

        mu_samples[it] = mu
        tau2_samples[it] = tau2
        sigma2_samples[it] = sigma2
        theta_samples[it] = theta

    return {
        "mu_samples": mu_samples,
        "tau2_samples": tau2_samples,
        "sigma2_samples": sigma2_samples,
        "theta_samples": theta_samples,
        "mu_mean": float(np.mean(mu_samples)),
        "tau2_mean": float(np.mean(tau2_samples)),
        "J": J,
    }


bhier = bayesian_hierarchical


def cheatsheet() -> str:
    return "bayesian_hierarchical({}) -> Bayesian hierarchical model (2-level normal)."
