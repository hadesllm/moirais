# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian survival model (exponential prior)."""

from __future__ import annotations

__all__ = ["bayesian_survival", "bsurv"]

from typing import Any, Union

import numpy as np
from scipy import stats


def bayesian_survival(
    times: Union[list, np.ndarray],
    events: Union[list, np.ndarray],
    *,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    prob: float = 0.95,
    n_iter: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Bayesian survival model with exponential likelihood and Gamma prior.

    Model: T_i ~ Exp(lambda)  (hazard lambda)
    Prior: lambda ~ Gamma(prior_a, prior_b)
    Posterior: lambda | data ~ Gamma(a + d, b + sum(t))

    where d = number of events and sum(t) = total observed time.
    For censored observations, the likelihood contribution is
    S(t) = exp(-lambda * t), leading to the same conjugate update.

    Parameters
    ----------
    times : array-like
        Observed times (n,).
    events : array-like
        Event indicators 0/1 (n,).
    prior_a : float
        Gamma prior shape.
    prior_b : float
        Gamma prior rate.
    prob : float
        Credible interval probability.
    n_iter : int
        Number of posterior samples to draw.
    seed : int
        Random seed.

    Returns
    -------
    dict
        posterior_mean : float -- posterior mean of hazard rate
        posterior_var : float
        ci_lower, ci_upper : float
        median_survival : float
        hazard_samples : ndarray

    References
    ----------
    Ibrahim, J. G., Chen, M.-H., & Sinha, D. (2001). *Bayesian
    Survival Analysis*, Springer, Ch. 3.
    """
    t = np.asarray(times, dtype=float).ravel()
    d = np.asarray(events, dtype=float).ravel()

    if len(t) == 0:
        raise ValueError("times must not be empty.")
    if len(t) != len(d):
        raise ValueError("times and events must have same length.")
    if np.any(t < 0):
        raise ValueError("times must be non-negative.")

    total_events = float(np.sum(d))
    total_time = float(np.sum(t))

    post_a = prior_a + total_events
    post_b = prior_b + total_time

    post_mean = post_a / post_b
    post_var = post_a / post_b ** 2

    ci_lo = float(stats.gamma.ppf((1 - prob) / 2, a=post_a, scale=1.0 / post_b))
    ci_hi = float(stats.gamma.ppf(1 - (1 - prob) / 2, a=post_a, scale=1.0 / post_b))

    rng = np.random.default_rng(seed)
    hazard_samples = rng.gamma(post_a, 1.0 / post_b, size=n_iter)
    median_survival = float(np.median(np.log(2) / hazard_samples))

    return {
        "posterior_mean": float(post_mean),
        "posterior_var": float(post_var),
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "median_survival": median_survival,
        "hazard_samples": hazard_samples,
        "posterior_a": float(post_a),
        "posterior_b": float(post_b),
    }


bsurv = bayesian_survival


def cheatsheet() -> str:
    return "bayesian_survival(times, events) -> Bayesian exponential survival model."
