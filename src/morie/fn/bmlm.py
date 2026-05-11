# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian multilevel (hierarchical) model."""

from __future__ import annotations

__all__ = ["bayesian_multilevel", "bmlm"]

from typing import Any, Union

import numpy as np


def bayesian_multilevel(
    group_means: Union[list, np.ndarray],
    group_ses: Union[list, np.ndarray],
    *,
    prior_mu: float = 0.0,
    prior_tau: float = 10.0,
    n_iter: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Bayesian multilevel (random effects) model via Gibbs sampling.

    Model (Normal-Normal hierarchical):
      y_j | theta_j ~ N(theta_j, sigma_j^2)   (known sigma_j)
      theta_j | mu, tau ~ N(mu, tau^2)
      mu ~ N(prior_mu, prior_tau^2)
      tau ~ Half-Cauchy(0, 5)

    Uses Gibbs sampling with the conditional posteriors.

    Parameters
    ----------
    group_means : array-like
        Observed group means (J,).
    group_ses : array-like
        Known standard errors per group (J,).
    prior_mu : float
        Prior mean for the grand mean.
    prior_tau : float
        Prior scale for the grand mean.
    n_iter : int
        Number of Gibbs iterations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        grand_mean : float
        grand_mean_sd : float
        tau_mean : float
        theta_means : ndarray (J,)
        shrinkage : ndarray (J,) -- shrinkage factors per group
        mu_samples : ndarray
        tau_samples : ndarray

    References
    ----------
    Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed.,
    CRC Press, Ch. 5.
    """
    y = np.asarray(group_means, dtype=float)
    s = np.asarray(group_ses, dtype=float)
    J = len(y)
    if len(s) != J:
        raise ValueError("group_means and group_ses must have same length.")
    if np.any(s <= 0):
        raise ValueError("group_ses must be positive.")

    rng = np.random.default_rng(seed)
    mu_cur = float(np.mean(y))
    tau_cur = float(np.std(y))
    theta = y.copy()

    mu_samples = np.empty(n_iter)
    tau_samples = np.empty(n_iter)
    theta_samples = np.empty((n_iter, J))

    for i in range(n_iter):
        tau2 = tau_cur ** 2 + 1e-30
        for j in range(J):
            prec_data = 1.0 / s[j] ** 2
            prec_prior = 1.0 / tau2
            prec_post = prec_data + prec_prior
            mean_post = (prec_data * y[j] + prec_prior * mu_cur) / prec_post
            theta[j] = rng.normal(mean_post, 1.0 / np.sqrt(prec_post))

        prior_prec = 1.0 / prior_tau ** 2
        prec_theta = J / tau2
        mu_prec = prior_prec + prec_theta
        mu_mean = (prior_prec * prior_mu + prec_theta * np.mean(theta)) / mu_prec
        mu_cur = rng.normal(mu_mean, 1.0 / np.sqrt(mu_prec))

        ss = float(np.sum((theta - mu_cur) ** 2))
        post_shape = (J - 1) / 2.0 + 1.0
        post_scale = ss / 2.0 + 1.0
        tau_cur = np.sqrt(1.0 / rng.gamma(post_shape, 1.0 / post_scale))

        mu_samples[i] = mu_cur
        tau_samples[i] = tau_cur
        theta_samples[i] = theta

    burn = n_iter // 2
    shrinkage = np.zeros(J)
    tau_mean = float(np.mean(tau_samples[burn:]))
    for j in range(J):
        B_j = s[j] ** 2 / (s[j] ** 2 + tau_mean ** 2)
        shrinkage[j] = B_j

    return {
        "grand_mean": float(np.mean(mu_samples[burn:])),
        "grand_mean_sd": float(np.std(mu_samples[burn:])),
        "tau_mean": tau_mean,
        "tau_sd": float(np.std(tau_samples[burn:])),
        "theta_means": np.mean(theta_samples[burn:], axis=0),
        "shrinkage": shrinkage,
        "mu_samples": mu_samples,
        "tau_samples": tau_samples,
    }


bmlm = bayesian_multilevel


def cheatsheet() -> str:
    return "bayesian_multilevel(means, ses) -> Bayesian multilevel (hierarchical) model."
