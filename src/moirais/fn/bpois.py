# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian Poisson model (Gamma-Poisson conjugate)."""

from __future__ import annotations

__all__ = ["bayesian_poisson", "bpois"]

from typing import Any, Union

import numpy as np
from scipy import stats


def bayesian_poisson(
    counts: Union[int, list, np.ndarray],
    exposure: float = 1.0,
    *,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Gamma-Poisson conjugate analysis for a Poisson rate.

    Prior: lambda ~ Gamma(prior_a, prior_b)  (shape, rate)
    Likelihood: x_i ~ Poisson(lambda * exposure_i)

    Posterior: lambda | x ~ Gamma(a + sum(x), b + T)

    Parameters
    ----------
    counts : int or array-like
        Observed count(s).
    exposure : float
        Total person-time or observation period.
    prior_a : float
        Gamma prior shape parameter.
    prior_b : float
        Gamma prior rate parameter.
    prob : float
        Credible interval probability.

    Returns
    -------
    dict
        posterior_mean, posterior_var, ci_lower, ci_upper,
        posterior_a, posterior_b

    References
    ----------
    Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed.,
    CRC Press, Ch. 2.
    """
    c = np.atleast_1d(np.asarray(counts, dtype=float))
    total = float(np.sum(c))

    if prior_a <= 0 or prior_b <= 0:
        raise ValueError("prior_a and prior_b must be positive.")
    if exposure <= 0:
        raise ValueError("exposure must be positive.")

    post_a = prior_a + total
    post_b = prior_b + exposure
    post_mean = post_a / post_b
    post_var = post_a / post_b ** 2

    ci_lo = float(stats.gamma.ppf((1 - prob) / 2, a=post_a, scale=1 / post_b))
    ci_hi = float(stats.gamma.ppf(1 - (1 - prob) / 2, a=post_a, scale=1 / post_b))

    return {
        "posterior_mean": float(post_mean),
        "posterior_var": float(post_var),
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "posterior_a": float(post_a),
        "posterior_b": float(post_b),
    }


bpois = bayesian_poisson


def cheatsheet() -> str:
    return "bayesian_poisson(counts) -> Gamma-Poisson conjugate Bayesian analysis."
