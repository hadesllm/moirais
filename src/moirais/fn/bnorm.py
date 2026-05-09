# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian normal model (conjugate, known variance)."""

from __future__ import annotations

__all__ = ["bayesian_normal", "bnorm"]

from typing import Any, Union

import numpy as np
from scipy import stats


def bayesian_normal(
    data: Union[list, np.ndarray],
    *,
    prior_mu: float = 0.0,
    prior_sigma: float = 10.0,
    sigma_known: float = 1.0,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Normal-normal conjugate analysis for a mean (known variance).

    Prior: mu ~ N(prior_mu, prior_sigma^2)
    Likelihood: x_i | mu ~ N(mu, sigma_known^2)

    Posterior: mu | x ~ N(post_mu, post_sigma^2) where

    .. math::

        \\tau_{\\text{post}} = \\tau_{\\text{prior}} + n \\tau_{\\text{data}}

        \\mu_{\\text{post}} = \\frac{\\tau_{\\text{prior}} \\mu_0
        + n \\tau_{\\text{data}} \\bar{x}}{\\tau_{\\text{post}}}

    Parameters
    ----------
    data : array-like
        Observed data.
    prior_mu : float
        Prior mean.
    prior_sigma : float
        Prior standard deviation for the mean.
    sigma_known : float
        Known data standard deviation.
    prob : float
        Credible interval probability.

    Returns
    -------
    dict
        posterior_mean, posterior_sd, ci_lower, ci_upper, n,
        prior_weight, data_weight

    References
    ----------
    Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed.,
    CRC Press, Ch. 2.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("Data must not be empty.")
    if prior_sigma <= 0 or sigma_known <= 0:
        raise ValueError("Standard deviations must be positive.")

    tau_prior = 1.0 / prior_sigma ** 2
    tau_data = n / sigma_known ** 2
    tau_post = tau_prior + tau_data

    post_mu = (tau_prior * prior_mu + tau_data * np.mean(x)) / tau_post
    post_sigma = 1.0 / np.sqrt(tau_post)

    z = stats.norm.ppf(1.0 - (1.0 - prob) / 2.0)
    ci_lo = post_mu - z * post_sigma
    ci_hi = post_mu + z * post_sigma

    return {
        "posterior_mean": float(post_mu),
        "posterior_sd": float(post_sigma),
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "n": n,
        "prior_weight": float(tau_prior / tau_post),
        "data_weight": float(tau_data / tau_post),
    }


bnorm = bayesian_normal


def cheatsheet() -> str:
    return "bayesian_normal(data) -> Normal-normal conjugate Bayesian analysis."
