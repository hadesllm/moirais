# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Conjugate posterior updating for standard models."""

from __future__ import annotations

import math
from typing import Any, Union

import numpy as np


def conjugate_posterior(
    data: Union[list, np.ndarray],
    *,
    model: str = "normal",
    prior_params: dict[str, float] | None = None,
    known_var: float = 1.0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compute the conjugate posterior distribution for standard models.

    Supported models:

    * ``"normal"`` -- Normal-Normal (known variance).
      Prior: N(prior_mu, prior_var). Likelihood: N(mu, known_var/n).
      Posterior: N(post_mu, post_var).

    * ``"beta_binomial"`` -- Beta-Binomial.
      Prior: Beta(a, b). Data: array of 0/1 outcomes.
      Posterior: Beta(a + sum(data), b + n - sum(data)).

    :param data: Observed data (array of floats or 0/1).
    :param model: ``"normal"`` or ``"beta_binomial"``.
    :param prior_params: Prior hyperparameters.
        For normal: ``{"mu": 0.0, "var": 1.0}``.
        For beta_binomial: ``{"a": 1.0, "b": 1.0}``.
    :param known_var: Known population variance (normal model only).
    :param alpha: Significance level for credible interval.
    :return: Dictionary with posterior_params, posterior_mean, posterior_sd,
        credible_interval.
    :raises ValueError: If model is not recognised or data is empty.

    References
    ----------
    Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A.,
    & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
    """
    arr = np.asarray(data, dtype=float)
    if len(arr) == 0:
        raise ValueError("data must not be empty.")

    from scipy import stats as _st

    if model == "normal":
        p = prior_params or {"mu": 0.0, "var": 1.0}
        prior_mu = float(p["mu"])
        prior_var = float(p["var"])
        n = len(arr)
        data_mean = float(np.mean(arr))
        likelihood_var = known_var / n
        post_var = 1.0 / (1.0 / prior_var + 1.0 / likelihood_var)
        post_mu = post_var * (prior_mu / prior_var + data_mean / likelihood_var)
        post_sd = math.sqrt(post_var)
        z = _st.norm.ppf(1.0 - alpha / 2.0)
        ci = (post_mu - z * post_sd, post_mu + z * post_sd)
        return {
            "posterior_params": {"mu": post_mu, "var": post_var},
            "posterior_mean": post_mu,
            "posterior_sd": post_sd,
            "credible_interval": ci,
            "model": "normal-normal (known variance)",
        }
    elif model == "beta_binomial":
        p = prior_params or {"a": 1.0, "b": 1.0}
        a_prior = float(p["a"])
        b_prior = float(p["b"])
        s = float(np.sum(arr))
        n = len(arr)
        a_post = a_prior + s
        b_post = b_prior + (n - s)
        post_mean = a_post / (a_post + b_post)
        post_var = (a_post * b_post) / ((a_post + b_post) ** 2 * (a_post + b_post + 1))
        post_sd = math.sqrt(post_var)
        ci_lo, ci_hi = _st.beta.ppf(alpha / 2, a_post, b_post), _st.beta.ppf(1 - alpha / 2, a_post, b_post)
        return {
            "posterior_params": {"a": a_post, "b": b_post},
            "posterior_mean": post_mean,
            "posterior_sd": post_sd,
            "credible_interval": (float(ci_lo), float(ci_hi)),
            "model": "beta-binomial",
        }
    else:
        raise ValueError(f"Unknown model '{model}'. Use 'normal' or 'beta_binomial'.")


bpost = conjugate_posterior


def cheatsheet() -> str:
    return "conjugate_posterior({}) -> Conjugate posterior updating for standard models."
