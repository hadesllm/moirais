# morie.fn — function file (hadesllm/morie)
"""Gibbs sampler for bivariate normal."""

from __future__ import annotations

__all__ = ["gibbs_bivariate_normal", "gibbs"]

from typing import Any

import numpy as np


def gibbs_bivariate_normal(
    mu: tuple[float, float] = (0.0, 0.0),
    sigma: tuple[float, float] = (1.0, 1.0),
    rho: float = 0.5,
    *,
    n_iter: int = 5000,
    burn_in: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    r"""
    Gibbs sampler for a bivariate normal distribution.

    Samples from BVN(mu, Sigma) using the full conditional
    distributions, which are univariate normals:

    .. math::

        X_1 | X_2 \\sim N\\bigl(\\mu_1 + \\rho \\frac{\\sigma_1}{\\sigma_2}
        (x_2 - \\mu_2),\\; \\sigma_1^2 (1 - \\rho^2)\\bigr)

    Parameters
    ----------
    mu : tuple of float
        Marginal means (mu1, mu2).
    sigma : tuple of float
        Marginal standard deviations (sigma1, sigma2).
    rho : float
        Correlation coefficient, must be in (-1, 1).
    n_iter : int
        Total number of iterations (including burn-in).
    burn_in : int
        Samples to discard.
    seed : int
        Random seed.

    Returns
    -------
    dict
        samples : ndarray (n_kept, 2)
        sample_mean : ndarray (2,)
        sample_corr : float

    Raises
    ------
    ValueError
        If |rho| >= 1 or sigmas are non-positive.

    References
    ----------
    Casella, G. & George, E. I. (1992). Explaining the Gibbs sampler.
    *The American Statistician*, 46(3), 167--174.
    Geman, S. & Geman, D. (1984). IEEE Trans. PAMI, 6(6), 721--741.
    """
    if abs(rho) >= 1.0:
        raise ValueError("rho must be in (-1, 1).")
    if sigma[0] <= 0 or sigma[1] <= 0:
        raise ValueError("sigma values must be positive.")
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1.")
    if burn_in >= n_iter:
        raise ValueError("burn_in must be < n_iter.")

    rng = np.random.default_rng(seed)
    s1, s2 = sigma
    m1, m2 = mu
    cond_sd1 = s1 * np.sqrt(1 - rho ** 2)
    cond_sd2 = s2 * np.sqrt(1 - rho ** 2)

    all_samples = np.empty((n_iter, 2))
    x1 = m1
    x2 = m2

    for i in range(n_iter):
        cond_mean1 = m1 + rho * (s1 / s2) * (x2 - m2)
        x1 = rng.normal(cond_mean1, cond_sd1)

        cond_mean2 = m2 + rho * (s2 / s1) * (x1 - m1)
        x2 = rng.normal(cond_mean2, cond_sd2)

        all_samples[i] = [x1, x2]

    kept = all_samples[burn_in:]
    corr_mat = np.corrcoef(kept[:, 0], kept[:, 1])

    return {
        "samples": kept,
        "sample_mean": np.mean(kept, axis=0),
        "sample_corr": float(corr_mat[0, 1]),
        "n_iter": n_iter,
        "burn_in": burn_in,
    }


gibbs = gibbs_bivariate_normal


def cheatsheet() -> str:
    return "gibbs_bivariate_normal(mu, sigma, rho) -> Gibbs sampler for bivariate normal."
