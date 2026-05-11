# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Gibbs sampler for normal mean and variance."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gibbs_normal(
    data: np.ndarray | list,
    *,
    n_iter: int = 5000,
    mu_0: float = 0.0,
    kappa_0: float = 1.0,
    alpha_0: float = 1.0,
    beta_0: float = 1.0,
    seed: int | None = None,
) -> DescriptiveResult:
    """
    Gibbs sampler for Normal(mu, sigma^2) with conjugate priors.

    Prior: mu | sigma^2 ~ N(mu_0, sigma^2/kappa_0),
           sigma^2 ~ InvGamma(alpha_0, beta_0).

    Parameters
    ----------
    data : array-like
        Observed data.
    n_iter : int
        Number of Gibbs iterations.
    mu_0, kappa_0 : float
        Normal prior hyperparameters.
    alpha_0, beta_0 : float
        Inverse-gamma prior hyperparameters.
    seed : int, optional
        Random seed.

    Returns
    -------
    DescriptiveResult
        extra has 'mu_samples', 'sigma2_samples'.

    References
    ----------
    Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed.
    CRC Press, Ch. 3.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(data, dtype=float)
    n = len(x)
    x_bar = np.mean(x)

    mu_samples = np.zeros(n_iter)
    sigma2_samples = np.zeros(n_iter)
    sigma2 = float(np.var(x, ddof=1))

    for i in range(n_iter):
        kappa_n = kappa_0 + n
        mu_n = (kappa_0 * mu_0 + n * x_bar) / kappa_n
        mu = rng.normal(mu_n, np.sqrt(sigma2 / kappa_n))
        mu_samples[i] = mu

        alpha_n = alpha_0 + n / 2
        beta_n = beta_0 + 0.5 * np.sum((x - mu) ** 2) + kappa_0 * n * (x_bar - mu_0) ** 2 / (2 * kappa_n)
        sigma2 = 1.0 / rng.gamma(alpha_n, 1.0 / beta_n)
        sigma2_samples[i] = sigma2

    burn = n_iter // 2
    return DescriptiveResult(
        name="gibbs_normal",
        value=float(np.mean(mu_samples[burn:])),
        extra={
            "mu_samples": mu_samples,
            "sigma2_samples": sigma2_samples,
            "posterior_mu_mean": float(np.mean(mu_samples[burn:])),
            "posterior_sigma2_mean": float(np.mean(sigma2_samples[burn:])),
        },
    )


bgibbs = gibbs_normal


def cheatsheet() -> str:
    return "gibbs_normal({}) -> Gibbs sampler for normal mean and variance."
