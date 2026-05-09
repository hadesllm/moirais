# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian predictive distribution (normal model)."""

from __future__ import annotations

__all__ = ["bayesian_predictive", "bpred"]

from typing import Any, Union

import numpy as np
from scipy import stats


def bayesian_predictive(
    data: Union[list, np.ndarray],
    *,
    prior_mu: float = 0.0,
    prior_kappa: float = 1.0,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    prob: float = 0.95,
    n_draws: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Bayesian posterior predictive distribution for a Normal model.

    Using the Normal-Inverse-Gamma conjugate model:
      x_i | mu, sigma^2 ~ N(mu, sigma^2)
      mu | sigma^2 ~ N(prior_mu, sigma^2 / prior_kappa)
      sigma^2 ~ Inv-Gamma(prior_alpha, prior_beta)

    The posterior predictive for a new observation x_new is
    a Student-t distribution:

    .. math::

        x_{\\text{new}} | \\text{data} \\sim t_{2\\alpha_n}
        \\left(\\mu_n, \\frac{\\beta_n (\\kappa_n + 1)}{\\alpha_n \\kappa_n}\\right)

    Parameters
    ----------
    data : array-like
        Observed data.
    prior_mu : float
        Prior mean.
    prior_kappa : float
        Prior precision scaling.
    prior_alpha : float
        Prior shape for inverse-gamma on variance.
    prior_beta : float
        Prior scale for inverse-gamma on variance.
    prob : float
        Predictive interval probability.
    n_draws : int
        Number of predictive samples to draw.
    seed : int
        Random seed.

    Returns
    -------
    dict
        predictive_mean, predictive_sd, pi_lower, pi_upper,
        predictive_df, predictive_samples

    References
    ----------
    Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed.,
    CRC Press, Ch. 3.
    Murphy, K. P. (2007). Conjugate Bayesian analysis of the Gaussian
    distribution. Technical Report, UBC.
    """
    x = np.asarray(data, dtype=float).ravel()
    n = len(x)
    if n == 0:
        raise ValueError("data must not be empty.")

    x_bar = float(np.mean(x))
    s2 = float(np.var(x, ddof=0)) if n > 1 else 0.0

    kappa_n = prior_kappa + n
    mu_n = (prior_kappa * prior_mu + n * x_bar) / kappa_n
    alpha_n = prior_alpha + n / 2.0
    beta_n = prior_beta + 0.5 * n * s2 + 0.5 * (prior_kappa * n * (x_bar - prior_mu) ** 2) / kappa_n

    df = 2.0 * alpha_n
    scale = np.sqrt(beta_n * (kappa_n + 1) / (alpha_n * kappa_n))

    rng = np.random.default_rng(seed)
    predictive_samples = mu_n + scale * rng.standard_t(df, size=n_draws)

    pi_lo = float(stats.t.ppf((1 - prob) / 2, df=df, loc=mu_n, scale=scale))
    pi_hi = float(stats.t.ppf(1 - (1 - prob) / 2, df=df, loc=mu_n, scale=scale))

    return {
        "predictive_mean": float(mu_n),
        "predictive_sd": float(scale * np.sqrt(df / (df - 2))) if df > 2 else float("inf"),
        "pi_lower": pi_lo,
        "pi_upper": pi_hi,
        "predictive_df": float(df),
        "predictive_samples": predictive_samples,
    }


bpred = bayesian_predictive


def cheatsheet() -> str:
    return "bayesian_predictive(data) -> Bayesian posterior predictive distribution."
